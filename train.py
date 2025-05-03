import torch
import os
import logging
import random
import argparse
import json
import glob

import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from dataset import LLVCDataset as Dataset
from model import Net
from torch.nn.parallel import DistributedDataParallel as DDP
import utils
import fairseq
import argparse
from fairseq.data.dictionary import Dictionary
import torch.serialization

torch.serialization.add_safe_globals([Dictionary]) #py보안 정책땜에 필요


#분산 학습에서 마스터 프로세스(master process) 가 동작하는 주소를 지정//
os.environ['MASTER_ADDR'] = 'localhost'
#마스터 프로세스가 통신할 때 사용할 포트 번호를 지정
os.environ['MASTER_PORT'] = '12355'
# check if port is available


def net_g_step(batch, net_g, device, fp16_run):
    # 배치 데이터를 받아 모델 net_g에 통과시키는 forward 연산을 수행하는 함수
    # fp16_run이 True이면 자동 mixed precision(autocast) 모드로 실행됨

    og, gt, target_index = batch  # batch에서 입력(og)과 정답(gt)을 분리

    og = og.to(device=device, non_blocking=True)  # 입력 데이터를 지정한 디바이스(GPU 또는 CPU)로 비동기 전송
    gt = gt.to(device=device, non_blocking=True)  # 정답 데이터도 동일하게 디바이스로 전송
    target_index = target_index.to(device=device)

    #output에 타겟 인덱스도 함께 넘겨주기
    with autocast(enabled=fp16_run):  # fp16_run이 True이면 float16 혼합 정밀도로 연산 수행
        output = net_g(og, target_index=target_index)  # 모델 net_g에 입력 og를 넣어 출력값 계산

    return output, gt, og  # 출력 결과, 정답, 입력 데이터를 함께 반환



def training_runner(rank, world_size, config, training_dir):
    log_dir = os.path.join(training_dir, "logs")  # 로그 디렉토리 경로 설정
    checkpoint_dir = os.path.join(training_dir, "checkpoints")  # 체크포인트 저장 디렉토리 경로 설정

    device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 사용 가능 여부 확인

    is_multi_process = world_size > 1  # 분산 학습 여부
    is_main_process = rank == 0  # 현재 프로세스가 메인인지 여부

    if is_main_process:
        os.makedirs(log_dir, exist_ok=True)  # 로그 디렉토리 생성
        os.makedirs(checkpoint_dir, exist_ok=True)  # 체크포인트 디렉토리 생성
        writer = SummaryWriter(log_dir=log_dir)  # TensorBoard 로그 기록기 초기화

    # dist.init_process_group(
    #     backend="gloo", init_method="env://", rank=rank, world_size=world_size
    # )  # PyTorch 분산 학습 프로세스 그룹 초기화 (환경변수 방식 사용)

    if world_size > 1:
        dist.init_process_group(backend="gloo", init_method="env://", rank=rank, world_size=world_size)

    if is_multi_process:
        torch.cuda.set_device(rank)  # rank에 해당하는 GPU 사용 설정

    torch.manual_seed(config['seed'])  # 랜덤 시드 고정

    data_train = Dataset(**config['data'], dset='train')  # 학습용 데이터셋 생성
    data_val = Dataset(**config['data'], dset='val')  # 검증용 데이터셋
    data_dev = Dataset(**config['data'], dset='dev')  # 개발용(예측 확인용) 데이터셋

    for ds in [data_train, data_val, data_dev]:
        logging.info(f"Loaded dataset at {ds.dset} containing {len(ds)} elements")  # 로드된 데이터셋 정보 로그

    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=config['batch_size'],
                                               shuffle=True)  # 학습용 데이터로더

    val_loader = torch.utils.data.DataLoader(data_val,
                                             batch_size=config['eval_batch_size'])  # 검증용 데이터로더

    dev_loader = torch.utils.data.DataLoader(data_dev,
                                             batch_size=config['eval_batch_size'])  # 개발용 데이터로더

    net_g = Net(**config['model_params'])  # 생성자(generator) 모델 초기화
    logging.info(f"Model size: {utils.model_size(net_g)}M params")  # 모델 파라미터 수 로그

    if is_multi_process:
        net_g = net_g.cuda(rank)  # 분산 학습일 경우 지정된 GPU로 이동
    else:
        net_g = net_g.to(device=device)  # 단일 GPU 또는 CPU 사용

    if config['discriminator'] == 'hfg':  # hfg 구조 선택 시
        from hfg_disc import ComboDisc, discriminator_loss, generator_loss, feature_loss
        net_d = ComboDisc()
    else:  # 일반 다중 주기 판별자 사용
        from discriminators import MultiPeriodDiscriminator, discriminator_loss, generator_loss, feature_loss
        net_d = MultiPeriodDiscriminator(periods=config['periods'])
    
    if is_multi_process:
        net_d = net_d.cuda(rank)  # 멀티 프로세스일 경우, 현재 rank에 해당하는 GPU로 판별자(net_d) 이동
    else:
        net_d = net_d.to(device=device)  # 싱글 프로세스일 경우 일반적인 device로 이동 (GPU 또는 CPU)

    optim_g = torch.optim.AdamW(
        net_g.parameters(),                     # 생성자(net_g)의 파라미터들에 대해
        config['optim']['lr'],                  # 학습률 설정
        betas=config['optim']['betas'],         # Adam의 beta 값 설정 (momentum 계수)
        eps=config['optim']['eps'],             # 수치 안정성 위한 작은 값
        weight_decay=config['optim']['weight_decay']  # 가중치 감소 (L2 정규화)
    )

    optim_d = torch.optim.AdamW(
        net_d.parameters(),                     # 판별자(net_d)의 파라미터들에 대해
        config['optim']['lr'],
        betas=config['optim']['betas'],
        eps=config['optim']['eps'],
        weight_decay=config['optim']['weight_decay']
    )


    if is_multi_process:
        net_g = DDP(net_g, device_ids=[rank])  # 생성자 모델을 분산 처리로 래핑
        net_d = DDP(net_d, device_ids=[rank])  # 판별자 모델도 마찬가지

    last_d_state = utils.latest_checkpoint_path(checkpoint_dir, "D_*.pth")  # 가장 최근의 판별자 체크포인트 경로 가져오기
    last_g_state = utils.latest_checkpoint_path(checkpoint_dir, "G_*.pth")  # 가장 최근의 생성자 체크포인트 경로 가져오기


    if last_d_state and last_g_state:  # 두 체크포인트가 모두 존재할 경우
        net_d, optim_d, lr, epoch, step = utils.load_checkpoint(
            last_d_state, net_d, optim_d)  # 판별자 체크포인트 로드

        net_g, optim_g, lr, epoch, step = utils.load_checkpoint(
            last_g_state, net_g, optim_g)  # 생성자 체크포인트 로드

        global_step = step  # 로드한 글로벌 스텝 수로 이어서 학습
        logging.info("Loaded generator checkpoint from %s" % last_g_state)  # 로드 로그 출력
        logging.info("Loaded discriminator checkpoint from %s" % last_d_state)
        logging.info("Generator learning rates restored to:" +
                    utils.format_lr_info(optim_g))  # 학습률 정보 출력
        logging.info("Discriminator learning rates restored to:" +
                    utils.format_lr_info(optim_d))
    else:  # 체크포인트가 없을 경우 학습 처음부터 시작
        lr = config['optim']['lr']  # 초기 학습률 설정
        global_step = 0             # 스텝 초기화
        epoch = 0                   # 에폭 초기화
    
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=config['lr_sched']['lr_decay']  # 생성자 학습률 스케줄러 (지수 감소)
    )

    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=config['lr_sched']['lr_decay']  # 판별자 학습률 스케줄러 (지수 감소)
    )

    scaler = GradScaler(enabled=config['fp16_run'])  # 혼합 정밀도(fp16) 학습 시 사용할 그래디언트 스케일러 설정

    # load fairseq model
    if config['aux_fairseq']['c'] > 0:  # fairseq 보조 손실 계수가 0보다 크면 (즉, 사용할 경우)
        cp_path = config['aux_fairseq']['checkpoint_path']  # fairseq 모델의 체크포인트 경로를 설정
        fairseq_model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([
            cp_path])  # 체크포인트에서 모델, 설정, 작업(task) 불러오기
        fairseq_model = fairseq_model[0]  # 여러 모델 중 첫 번째 모델 선택
        fairseq_model.eval().to(device)  # 평가 모드로 전환하고 GPU 또는 CPU로 이동
    else:
        fairseq_model = None  # 사용할 모델이 없을 경우 None으로 설정


    cache = [] # 학습 데이터를 캐시할 리스트 (이후 반복 학습 시 재사용)
    loss_mel_avg = utils.RunningAvg() # mel 스펙트로그램 손실의 이동 평균을 계산하기 위한 도우미 클래스
    loss_fairseq_avg = utils.RunningAvg()  # fairseq 기반 손실의 이동 평균 저장용 클래스

    for epoch in range(epoch, 10000):  # 현재 에폭부터 시작해서 최대 10000 에폭까지 반복
        # train_loader.batch_sampler.set_epoch(epoch)  # (분산 학습 시) 데이터 셔플링을 위해 에폭마다 시드 설정, 현재는 주석 처리됨

        net_g.train()  # 생성자 모델을 학습 모드로 전환
        net_d.train()  # 판별자 모델을 학습 모드로 전환

        use_cache = len(cache) == len(train_loader)  # 캐시된 배치가 전체 train_loader 크기와 같으면 캐시 사용
        data = cache if use_cache else enumerate(train_loader)  # 캐시된 데이터를 사용할지 새로 로드할지 결정

        if is_main_process:
            lr = optim_g.param_groups[0]["lr"]  # 메인 프로세스일 경우 현재 생성자 학습률 가져오기

        # 다음 체크포인트까지 남은 스텝을 보여주는 진행 표시줄 초기화
        progress_bar = tqdm(range(config['checkpoint_interval']))  
        progress_bar.update(global_step % config['checkpoint_interval'])  # 현재 스텝을 기준으로 진행 표시줄 위치 맞추기

        for batch_idx, batch in data:  # 학습 데이터를 한 배치씩 반복
            output, gt, og = net_g_step(batch, net_g, device, config['fp16_run'])  
            # net_g_step을 통해 생성자 모델(net_g)에 입력 og를 넣고 예측 output을 얻음
            # gt: 정답 오디오, og: 입력 오디오

            # 입력과 출력 오디오에서 랜덤한 길이(segment_size) 만큼 잘라서 사용
            if config['segment_size'] < output.shape[-1]:  # 출력 길이가 segment_size보다 길 경우
                start_idx = random.randint(  # 랜덤한 시작 위치 선택
                    0, output.shape[-1] - config['segment_size'] - 1)
                gt_sliced = gt[:, :, start_idx:start_idx + config['segment_size']]  # 정답 오디오 잘라내기
                output_sliced = output.detach()[:, :, start_idx:start_idx + config['segment_size']]  # 생성된 오디오도 잘라냄 (no grad)
            else:
                gt_sliced = gt  # segment_size보다 짧으면 그냥 전체 사용
                output_sliced = output.detach()

            with autocast(enabled=config['fp16_run']):  # 혼합 정밀도(fp16) 사용 여부에 따라 자동 캐스팅
                # 판별자 모델에 sliced 오디오를 넣어 진짜/가짜 판단
                y_d_hat_r, y_d_hat_g, _, _ = net_d(output_sliced, gt_sliced)

                with autocast(enabled=False):  # 손실 계산은 float32로 안정적으로 수행
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                        y_d_hat_r, y_d_hat_g
                    )  # 진짜/가짜 판별 결과를 통해 판별자 손실 계산

            optim_d.zero_grad()  # 판별자 optimizer의 기울기 초기화
            scaler.scale(loss_disc).backward()  # fp16 scaler로 손실 역전파
            scaler.unscale_(optim_d)  # optimizer에 적용되기 전, 스케일을 되돌림

            if config['grad_clip_threshold'] is not None:  # gradient clipping 임계값이 설정되어 있으면 norm 기준 클리핑
                grad_norm_d = torch.nn.utils.clip_grad_norm_(
                    net_d.parameters(), config['grad_clip_threshold'])

            grad_norm_d = utils.clip_grad_value_(
                net_d.parameters(), config['grad_clip_value'])  # gradient 값을 기준으로 clipping 수행
            scaler.step(optim_d)  # scaler를 통해 optimizer step 수행 (파라미터 업데이트)

            with autocast(enabled=config['fp16_run']):  # 혼합 정밀도(fp16)를 사용하는 경우에만 자동 캐스팅
                # Generator 단계
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(gt, output)  
                # 정답(gt)과 생성된 오디오(output)를 판별자에 넣어서 진짜/가짜 판단과 feature map 추출

                if fairseq_model is not None:  # fairseq 모델이 있을 경우
                    loss_fairseq = utils.fairseq_loss(
                        output, gt, fairseq_model) * config['aux_fairseq']['c']  
                    # fairseq 기반 auxiliary 손실 계산 후 계수(c) 곱함
                else:
                    loss_fairseq = torch.tensor(0.0)  # 사용할 모델이 없으면 손실 0
                loss_fairseq_avg.update(loss_fairseq)  # fairseq 손실 이동 평균 업데이트

                with autocast(enabled=False):  # 정밀한 손실 계산을 위해 float32로 수행
                    if config['aux_mel']['c'] > 0:  # mel 손실 사용 여부
                        loss_mel = utils.aux_mel_loss(
                            output, gt, config) * config['aux_mel']['c']  # mel 손실 계산 및 계수 곱함
                    else:
                        loss_mel = torch.tensor(0.0)  # 사용하지 않으면 0
                    loss_mel_avg.update(loss_mel)  # mel 손실 이동 평균 업데이트

                    loss_fm = feature_loss(fmap_r, fmap_g) * config['feature_loss_c']  # Feature matching 손실 계산
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)  # 생성자에 대한 판별자 손실 (가짜라고 판단받은 정도)
                    loss_gen = loss_gen * config['disc_loss_c']  # 생성자 손실에 계수 곱함

                    # 전체 생성자 손실 계산 (총합 = 판별 손실 + FM + mel + fairseq)
                    loss_gen_all = (loss_gen + loss_fm) + loss_mel + loss_fairseq

            optim_g.zero_grad()  # 생성자 optimizer의 기울기 초기화
            scaler.scale(loss_gen_all).backward()  # 전체 손실에 대해 fp16 scaler로 역전파 수행
            scaler.unscale_(optim_g)  # 스케일링 된 기울기를 원래대로 되돌림

            if config['grad_clip_threshold'] is not None:  # grad clipping 임계값이 설정되어 있을 경우
                grad_norm_g = torch.nn.utils.clip_grad_norm_(
                    net_g.parameters(), config['grad_clip_threshold'])  # norm 기준으로 clipping

            grad_norm_g = utils.clip_grad_value_(
                net_g.parameters(), config['grad_clip_value'])  # 값 기준으로 clipping 수행

            scaler.step(optim_g)  # 생성자 optimizer로 파라미터 업데이트 수행
            scaler.update()  # GradScaler 내부 상태 업데이트 (스케일값 조정)

            global_step += 1  # 전체 학습 스텝 증가
            progress_bar.update(1)  # tqdm 진행바 한 칸 업데이트

            if is_main_process and global_step > 0 and (global_step % config['log_interval'] == 0):
                # 메인 프로세스이고, global_step이 log_interval의 배수일 때 로그 및 오디오 기록을 수행

                lr = optim_g.param_groups[0]["lr"]  # 현재 생성자(Generator)의 학습률을 가져옴

                if loss_mel > 50:
                    loss_mel = 50  # TensorBoard 시각화를 위해 mel 손실값이 너무 크면 50으로 클리핑

                # TensorBoard에 기록할 스칼라(수치형) 로그 저장용 딕셔너리 생성
                scalar_dict = {
                    "loss/g/total": loss_gen_all,       # 생성자 전체 손실
                    "loss/d/total": loss_disc,          # 판별자 전체 손실
                    "learning_rate": lr,                # 현재 학습률
                    "grad_norm_d": grad_norm_d,         # 판별자의 gradient norm
                    "grad_norm_g": grad_norm_g,         # 생성자의 gradient norm
                }

                # mel 손실 및 feature matching 손실도 추가 기록
                scalar_dict.update({
                    "loss/g/fm": loss_fm,               # feature matching 손실
                    "loss/g/mel": loss_mel,             # mel 손실
                })

                # mel auxiliary 손실 평균 기록 (사용 중일 경우)
                if config['aux_mel']['c'] > 0:
                    scalar_dict.update({"train_metrics/mel": loss_mel_avg()})  # 이동 평균값 저장
                    loss_mel_avg.reset()  # 평균 초기화

                # fairseq auxiliary 손실도 기록 (모델이 있을 경우)
                if fairseq_model is not None:
                    scalar_dict.update({
                        "loss/g/fairseq": loss_fairseq,  # 현재 step의 fairseq 손실
                    })
                    scalar_dict.update({
                        "train_metrics/fairseq": loss_fairseq_avg()  # 이동 평균값
                    })
                    loss_fairseq_avg.reset()  # 평균 초기화

                # 생성자 내부 세부 손실 항목 나열 (리스트 형태로 여러 손실이 있을 수 있음)
                scalar_dict.update({
                    "loss/g/{}".format(i): v for i, v in enumerate(losses_gen)
                })

                # 판별자의 real loss 항목들 추가
                scalar_dict.update({
                    "loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)
                })

                # 판별자의 generated loss 항목들 추가
                scalar_dict.update({
                    "loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)
                })

                # 오디오 샘플 저장용 딕셔너리 생성
                audio_dict = {}

                # 정답 오디오를 TensorBoard에 기록 (최대 3개)
                audio_dict.update({
                    f"train_audio/gt_{i}": gt[i].data.cpu().numpy()
                    for i in range(min(3, gt.shape[0]))
                })

                # 입력 오디오를 TensorBoard에 기록
                audio_dict.update({
                    f"train_audio/in_{i}": og[i].data.cpu().numpy()
                    for i in range(min(3, og.shape[0]))
                })

                # 생성된 오디오를 TensorBoard에 기록
                audio_dict.update({
                    f"train_audio/pred_{i}": output[i].data.cpu().numpy()
                    for i in range(min(3, output.shape[0]))
                })

                net_g.eval()  # 이후 벤치마크 및 검증을 위해 모델을 평가 모드로 전환

                # 벤치마크 디렉토리에서 오디오 파일을 불러와 테스트 수행
                test_wavs = [
                    (
                        os.path.basename(p),  # 파일 이름만 추출 (예: test01.wav)
                        utils.load_wav_to_torch(p, config['data']['sr']),  # 오디오 파일을 지정된 샘플레이트로 로드
                    )
                    for p in glob.glob(config['test_dir'] + "/*.wav")  # test_dir 안의 모든 .wav 파일 가져오기
                ]

                logging.info("Testing...")  # 테스트 시작 로그

                speaker_map = {
                    0: "conan",
                    1: "keroro",
                    2: "shinchan"
                }


                # 벤치마크 오디오들을 모델에 통과시켜 예측 결과 저장
                for test_wav_name, test_wav in tqdm(test_wavs, total=len(test_wavs)):
                    for idx in range(3):  # 인덱스 0, 1, 2 전부 변환
                        target_index = torch.tensor([idx], dtype=torch.long).to(device)
                        test_out = net_g(test_wav.unsqueeze(0).unsqueeze(0).to(device), target_index=target_index)
                        audio_dict.update({
                            f"test_audio/{test_wav_name}_{speaker_map[idx]}": test_out[0].data.cpu().numpy()
                        })

                # dev 및 val 검증용 데이터셋을 사용해 성능 평가
                for loader in [dev_loader, val_loader]:
                    loader_name = "dev" if loader == dev_loader else "val"  # 데이터셋 이름 결정
                    v_data = enumerate(loader)
                    logging.info(f"Validating on {loader_name} dataset...")

                    # 평가 지표 평균 계산 객체 초기화
                    v_loss_mel_avg = utils.RunningAvg()
                    v_loss_fairseq_avg = utils.RunningAvg()
                    v_mcd_avg = utils.RunningAvg()

                    with torch.no_grad():  # 평가 시 gradient 계산 비활성화
                        for v_batch_idx, v_batch in tqdm(v_data, total=len(loader)):
                            v_output, v_gt, og = net_g_step(v_batch, net_g, device, config['fp16_run'])  # 생성자 forward pass

                        if config['aux_mel']['c'] > 0:
                            v_loss_mel = utils.aux_mel_loss(output, gt, config) * config['aux_mel']['c']
                            v_loss_mel_avg.update(v_loss_mel)

                        if fairseq_model is not None:
                            with autocast(enabled=config['fp16_run']):
                                v_loss_fairseq = utils.fairseq_loss(output, gt, fairseq_model) * config['aux_fairseq']['c']
                                v_loss_fairseq_avg.update(v_loss_fairseq)

                        v_mcd = utils.mcd(v_output, v_gt, config['data']['sr'])  # Mel Cepstral Distortion 계산
                        v_mcd_avg.update(v_mcd)

                    # scalar_dict에 평가 지표 추가 기록
                    if config['aux_mel']['c'] > 0:
                        scalar_dict.update({
                            f"{loader_name}_metrics/mel": v_loss_mel_avg(),
                            f"{loader_name}_metrics/mcd": v_mcd_avg()
                        })
                        v_loss_mel_avg.reset()

                    if fairseq_model is not None:
                        scalar_dict.update({
                            f"{loader_name}_metrics/fairseq": v_loss_fairseq_avg()
                        })
                        v_loss_fairseq_avg.reset()

                    v_mcd_avg.reset()

                    # 평가용 오디오 샘플 3개씩 기록 (GT, 입력, 생성)
                    audio_dict.update({
                        f"{loader_name}_audio/gt_{i}": v_gt[i].data.cpu().numpy()
                        for i in range(min(3, v_gt.shape[0]))
                    })
                    audio_dict.update({
                        f"{loader_name}_audio/in_{i}": og[i].data.cpu().numpy()
                        for i in range(min(3, og.shape[0]))
                    })
                    audio_dict.update({
                        f"{loader_name}_audio/pred_{i}": v_output[i].data.cpu().numpy()
                        for i in range(min(3, v_output.shape[0]))
                    })

                net_g.train()  # 검증이 끝났으므로 모델을 다시 학습 모드로 전환

                # 준비된 스칼라/오디오 데이터를 TensorBoard에 기록
                utils.summarize(
                    writer=writer,  # TensorBoard SummaryWriter 객체
                    global_step=global_step,  # 현재 학습 스텝
                    scalars=scalar_dict,  # 손실, 학습률, 그래디언트 norm 등 스칼라 로그
                    audios=audio_dict,  # 입력/출력/정답 오디오 샘플들
                    audio_sampling_rate=config['data']['sr'],  # 오디오 샘플링 주파수
                )

                # 체크포인트 저장 타이밍이면 모델과 옵티마이저 상태 저장
                if global_step % config['checkpoint_interval'] == 0:
                    g_checkpoint = os.path.join(checkpoint_dir, f"G_{global_step}.pth")  # 생성자 체크포인트 경로
                    d_checkpoint = os.path.join(checkpoint_dir, f"D_{global_step}.pth")  # 판별자 체크포인트 경로

                    # 생성자 상태 저장
                    utils.save_state(
                        net_g,
                        optim_g,
                        lr,
                        epoch,
                        global_step,
                        g_checkpoint
                    )

                    # 판별자 상태 저장
                    utils.save_state(
                        net_d,
                        optim_d,
                        lr,
                        epoch,
                        global_step,
                        d_checkpoint
                    )

                    logging.info(f"Saved checkpoints to {g_checkpoint} and {d_checkpoint}")  # 로그 출력
                    progress_bar.reset()  # tqdm 진행바 초기화
                torch.cuda.empty_cache()# 불필요한 GPU 캐시 메모리 해제
        scheduler_g.step()# 생성자 학습률 스케줄러 업데이트
        scheduler_d.step()# 판별자 학습률 스케줄러 업데이트

    if is_main_process:
        logging.info("Training is done. The program is closed.")# 메인 프로세스에서 학습 종료 로그 출력

def train_model(gpus, config, training_dir):
    # 모델 학습을 실행하는 메인 함수. 여러 GPU를 활용한 멀티 프로세싱 학습을 지원 -> 싱글 gpu 사용으로 코드 수정

    deterministic = torch.backends.cudnn.deterministic  # 현재 cudnn의 deterministic 설정을 백업
    benchmark = torch.backends.cudnn.benchmark  # 현재 cudnn의 benchmark 설정을 백업

    # PREV_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", None)  # 기존 CUDA_VISIBLE_DEVICES 환경변수 저장

    # if PREV_CUDA_VISIBLE_DEVICES is None:
    #     PREV_CUDA_VISIBLE_DEVICES = None  # 이전 설정이 없었다면 None으로 유지
    #     os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    #         [str(gpu) for gpu in gpus])  # 학습에 사용할 GPU 목록을 환경변수로 설정 (예: "0,1,2")
    # else:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = PREV_CUDA_VISIBLE_DEVICES  # 기존에 설정된 값이 있으면 그대로 유지

    torch.backends.cudnn.deterministic = False  # 학습 속도를 위해 cudnn의 결정론적 모드를 끔
    torch.backends.cudnn.benchmark = False  # 입력 크기가 일정하지 않더라도 성능 최적화를 비활성화

    # mp.spawn(  # torch.multiprocessing.spawn: 멀티프로세싱 시작
    #     training_runner,  # 실행할 함수 (GPU 별 학습 루프)
    #     nprocs=len(gpus),  # 프로세스 수 = 사용 GPU 수
    #     args=(  # training_runner에 넘겨줄 인자들
    #         len(gpus),     # world_size (총 GPU 수)
    #         config,        # 학습 설정 딕셔너리
    #         training_dir   # 로그/체크포인트 저장 디렉토리
    #     )
    # )
        # 멀티 프로세싱 대신 단일 프로세스로 바로 실행
    training_runner(
        rank=0,            # GPU ID 0 사용
        world_size=1,      # 전체 GPU 수 1개로 간주
        config=config,
        training_dir=training_dir
    )

    # if PREV_CUDA_VISIBLE_DEVICES is None:
    #     del os.environ["CUDA_VISIBLE_DEVICES"]  # 학습 후, 임시로 설정했던 환경변수 삭제

    torch.backends.cudnn.deterministic = deterministic  # 이전 cudnn 설정 복원
    torch.backends.cudnn.benchmark = benchmark  # 이전 cudnn 설정 복원

def main():
    parser = argparse.ArgumentParser()  # 커맨드라인 인자를 파싱하기 위한 ArgumentParser 객체 생성
    parser.add_argument('--dir', "-d", type=str,
                        help="Path to save checkpoints and logs.")  # 저장 디렉토리 인자 추가
    args = parser.parse_args()  # 커맨드라인 인자 파싱
    args.dir = "./llvc_nc"

    with open(os.path.join(args.dir, "config.json")) as f:
        config = json.load(f)  # 지정된 디렉토리에서 config.json 파일을 열고, JSON 설정 불러오기

    gpus = [i for i in range(torch.cuda.device_count())]  # 현재 사용 가능한 모든 GPU 인덱스를 리스트로 생성
    logging.info("Using GPUs: {}".format(gpus))  # 어떤 GPU를 사용할지 로그 출력

    # fairseq 모델을 사용할 경우 사전학습된 체크포인트가 존재하는지 확인
    if config['aux_fairseq']['c'] > 0:
        if not os.path.exists(config['aux_fairseq']['checkpoint_path']):  # 지정된 경로에 체크포인트가 없으면
            print(f"Fairseq checkpoint not found at {config['aux_fairseq']['checkpoint_path']}")  # 경고 출력
            checkpoint_url = "https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt"  # 다운로드할 기본 URL
            print(f"Downloading from {checkpoint_url}")  # 다운로드 로그 출력

            checkpoint_folder = os.path.dirname(config['aux_fairseq']['checkpoint_path'])  # 저장 폴더 경로 추출
            if not os.path.exists(checkpoint_folder):
                os.makedirs(checkpoint_folder)  # 폴더가 없으면 생성

            os.system(f"wget {checkpoint_url} -P {checkpoint_folder}")  # wget 명령어로 체크포인트 다운로드

    train_model(gpus, config, args.dir)  # 모든 준비가 끝나면 train_model 함수 호출하여 학습 시작


if __name__ == "__main__":
    main()
