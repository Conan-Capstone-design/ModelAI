#A collection of useful helper functions
#오디오 신호 처리와 모델 학습 과정을 보조하는 기능 모아놓은 코드

import os                   # OS(운영체제)와의 상호작용을 위한 표준 라이브러리
import json                 # JSON 파일을 읽고 쓰는 데 사용
import torch                # PyTorch 딥러닝 프레임워크
import glob                 # 특정 패턴이 들어간 파일을 검색
import auraloss             # 오디오 신호 처리에 관련된 손실 함수를 제공
import librosa              # 오디오 신호 처리를 위한 라이브러리(오디오 로딩/변환)
import numpy as np          # 수치 계산을 위한 라이브러리
import torch.nn.functional as F        # PyTorch의 함수형 API(손실 함수, activation)
from torchaudio.transforms import MFCC  # torchaudio에서 제공하는 MFCC 변환 클래스를 임포트(음성 특징 추출)
from mel_processing import mel_spectrogram_torch    # mel 스펙트로그램 생성에 사용되는 함수


class Params():
    """
    JSON 파일에서 하이퍼파라미터를 로드하고, 이를 객체의 속성으로 사용하기 위한 클래스
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

#생성자: json_path에 있는 JSON 파일을 읽고 self.__dict__에 업데이트
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)   #JSON 파일을 파이썬 딕셔너리로 로드
            self.__dict__.update(params)    #현재 객체의 __dict__를 읽은 파라미터로 갱신

    def save(self, json_path):
            #현재 객체에 담긴 속성들을 json_path에 JSON 형태로 저장
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)       #JSON 파일을 들여쓰기 4칸으로 예쁘게 저장

    def update(self, json_path):
        """Loads parameters from json file"""
        #json_path에 있는 JSON 파일에서 파라미터를 다시 읽어와 객체 속성을 갱신
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        #params.dict 형태로 딕셔너리처럼 액세스 가능
        return self.__dict__


#dir_path 경로 안에서 특정 패턴(regex)에 매칭되는 파일들을 찾고, 그 중 가장 최근(숫자로 정렬했을 때 가장 큰 값)의 파일 경로를 반환
def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    filelist = glob.glob(os.path.join(dir_path, regex))     # dir_path 내에서 G_*.pth 패턴에 맞는 파일 목록을 가져옴
    if len(filelist) == 0:
        return None
    #정렬 기준: 파일 이름 중 숫자만 뽑아 정수화, 그 중 가장 큰 숫자의 파일이 최신
    filelist.sort(key=lambda f: int("".join(filter(str.isdigit, f))))       #파일명에 포함된 숫자를 추출
    filepath = filelist[-1]     #정렬 후 마지막(가장 큰 숫자) 파일 경로
    return filepath


"""
체크포인트(.pth) 파일을 로드해서 model과 optimizer의 상태를 복원원
checkpoint_path: 체크포인트 파일 경로
model: 로드할 모델
optimizer: 로드할 옵티마이저 (옵션)
load_opt: 옵티마이저도 로드할지 여부
"""
def load_checkpoint(checkpoint_path, model, optimizer=None, load_opt=True):
    # 체크포인트 파일이 실제 존재하는지 검증
    assert os.path.isfile(
        checkpoint_path), f"Checkpoint '{checkpoint_path}' not found"
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")       #체크포인트 파일을 로딩, CPU 메모리에 로드 가능

    #모델이 DataParallel 형태로 model.module을 감싸고 있으면
    if hasattr(model, "module"):        #내부의 실제 모델(model.module)에 상태를 로드
        model.module.load_state_dict(checkpoint_dict["model"])
    else:       #그냥 model에 로드
        model.load_state_dict(checkpoint_dict["model"])

    #체크포인트에 저장되어 있던 학습 상태 정보(epoch, step, learning_rate)를 불러옴
    epoch = checkpoint_dict["epoch"]
    step = checkpoint_dict["step"]
    learning_rate = checkpoint_dict["learning_rate"]
    #optimizer가 주어지고, load_opt=True면, 저장되어 있던 옵티마이저 상태(optimizer dict)도 복원
    if optimizer is not None and load_opt:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
        print(f"Loaded optimizer with learning rate {learning_rate}")
    print("Loaded checkpoint '{}' (epoch {}, step {})".format(
        checkpoint_path, epoch, step))
    return model, optimizer, learning_rate, epoch, step


#모델 상태를 저장하기 전에 로그를 출력(“몇 에폭에서 어느 파일 경로로 저장 중”)
def save_state(model, optimizer, learning_rate, epoch, step, checkpoint_path):
    print(
        "Saving model and optimizer state at epoch {} to {}".format(
            epoch, checkpoint_path
        )
    )
    #model이 DataParallel로 감싸져 있다면 .module로 실제 모델의 state_dict를 얻어옴
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

# 필요한 정보를 모두 딕셔너리로 묶어서 torch.save로 체크포인트를 저장
    torch.save(
        {
            "model": state_dict,
            "epoch": epoch,
            "step": step,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def model_size(model):
    """
    모델 파라미터(학습 가능한 텐서)의 총 개수를 백만 단위(M)로 환산하여 반환
    Returns size of the `model` in millions of parameters.
    """
    num_train_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    params_scaled = num_train_params / 1e6      #백만 단위(M)로 환산
    # round to 2 decimal places
    return round(params_scaled, 2)       #소수점 둘째 자리까지 반올림하여 반환


#옵옵티마이저의 param_groups 정보를 읽어, 각 그룹별 파라미터 개수와 학습률을 문자열로 구성해 반환
def format_lr_info(optimizer):
    lr_info = ""
    #파라미터 개수 / (1024^2)는 대략 메가 단위 파라미터로 추정
    for i, pg in enumerate(optimizer.param_groups):
        lr_info += " {group %d: params=%.5fM lr=%.1E}" % (
            i, sum([p.numel() for p in pg['params']]) / (1024 ** 2), pg['lr'])
    return lr_info

#주어진 파라미터들에 대해 기울기를 clip_value 범위 안으로 자름
#norm_type : 기울기 계산 시 사용
def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]   #parameters가 단일 텐서이면 리스트로 감싸기
    # p.grad가 존재하는 파라미터만 필터링
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    # 각 파라미터별로 norm을 측정하고, clip_value가 있으면 [-clip_value, clip_value] 범위로 자름
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm       #디버깅이나 로깅 목적으로 클리핑 후 전체 norm을 확인할 때 유용


#output과 gt(ground truth) 간의 멀티해상도 STFT 스펙트럼 차이를 구하는 함수
def multires_loss(output, gt, sr, params):
    loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
        scale="mel",           # mel 스케일 사용
        sample_rate=sr,        # 샘플레이트
        device=output.device,  # 텐서가 존재하는 디바이스로 맞춤
        **params               # 기타 파라미터를 모두 전달
    )
    return loss_fn(output, gt)      #출력 오디오(output)와 실제 오디오(gt) 사이의 STFT 손실값을 계산해 반환

#별도의 멜 스펙트럼 기반 손실을 계산.
#config에 따라 multires 모드 또는 rvc 모드를 적용
def aux_mel_loss(output, gt, config):
    sr = config['data']['sr']       #샘플레이트
    aux_mel_loss_type = config['aux_mel']['type']   # 어떤 방식의 멜 손실을 쓸지
    config_params = config['aux_mel']['params']     # 해당 손실에 필요한 파라미터들
    if aux_mel_loss_type == "multires":
        #멀티해상도 STFT 손실
        param_dict = {}     #STFT 관련 파라미터를 param_dict에 구성
        config_params = config['aux_mel']['params']
        param_dict['fft_sizes'] = config_params['n_fft']
        param_dict['hop_sizes'] = config_params['hop_size']
        param_dict['win_lengths'] = config_params['win_size']
        param_dict['n_bins'] = config_params['num_mels']
        return multires_loss(output, gt, sr, param_dict)
    
    elif aux_mel_loss_type == "rvc":
        # RVC(에서 사용하는 방식의 mel 스펙트럼) 손실
        param_dict = config_params
        for k in param_dict:
            #RVC 파라미터가 리스트인 경우 첫 번째 값을 사용
            if isinstance(param_dict[k], list):
                param_dict[k] = param_dict[k][0]
        # gt와 output에 대해 mel 스펙트로그램 생성
        gt_mel = mel_spectrogram_torch(
            gt.float().squeeze(1),
            sr,
            **param_dict
        )
        output_mel = mel_spectrogram_torch(
            output.float().squeeze(1),
            sr,
            **param_dict
        )
        # L1 Loss를 이용해 두 mel 스펙트로그램(GT와 output 오디오)의 차이를 계산
        loss_mel = F.l1_loss(
            output_mel, gt_mel)
        return loss_mel
    else:
        raise ValueError(f"Unknown aux mel loss type, {aux_mel_loss_type}")


#MCD(Mel Cepstral Distortion)를 직접 계산하는 함수
def mcd(predicted_audio, gt_audio, sr):
    mfcc = MFCC(sample_rate=sr).to(predicted_audio.device)
    #예측 오디오와 실제 오디오를 MFCC로 변환
    predicted_mfcc = mfcc(predicted_audio)
    gt_mfcc = mfcc(gt_audio.to(predicted_audio.device))
    return torch.mean(torch.abs(predicted_mfcc - gt_mfcc))      #두 MFCC의 절댓값 차이를 구해 평균을 낸 뒤 반환

def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sampling_rate=16000,
):
    #tensorboardX나 PyTorch의 SummaryWriter 등을 이용해 스칼라, 히스토그램, 이미지, 오디오 로그를 편리하게 기록할 수 있도록 정리
    # 스칼라(예: loss, accuracy 등)을 기록
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    # 히스토그램(예: 파라미터 분포 등)을 기록
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)

#누적 평균을 관리하는 클래스
class RunningAvg:
    def __init__(self):
        self.n = 0   # 지금까지 업데이트된 횟수
        self.avg = 0 # 현재까지의 평균

    def update(self, val):
        # 누적 평균을 계산: (이전 평균 * 이전 카운트 + 새 값) / (이전 카운트+1)
        self.avg = (self.avg * self.n + val) / (self.n + 1)
        self.n += 1

    def reset(self):
        # 평균 계산을 리셋
        self.n = 0
        self.avg = 0

    def __call__(self):
        # 객체를 함수처럼 호출하면 현재 평균을 반환
        return self.avg



#sr 샘플레이트로 resample하여 로드
def load_wav_to_torch(full_path, sr):
    data = librosa.load(full_path, sr=sr)[0]        #오디오 파일 불러옴, [0]: 실제 오디오 파형만 받아옴
    return torch.FloatTensor(data.astype(np.float32))       #np.float32로 변환 후 torch.FloatTensor로 만들어 PyTorch 텐서로 반환


"""
예측 음성과 실제 음성의 특징 벡터를 추출하여 이 둘 사이의 MSE 손실을 계산
'End-to-end Text to Speech' 등에 사용하는 방식과 유사한 예시
"""
def fairseq_loss(output, gt, fairseq_model):
    """
    fairseq feature mse loss, based on https://arxiv.org/abs/2301.04388
    """
    # [B, 1, T] -> [B, T]
    #squeeze(1) : (batch, 1, samples) 형태의 텐서를 (batch, samples)로 만드는 과정(모노 오디오 채널 차원 제거)
    gt = gt.squeeze(1)
    output = output.squeeze(1)
    # fairseq 모델에서 지원하는 feature 추출
    gt_f = fairseq_model.feature_extractor(gt)
    output_f = fairseq_model.feature_extractor(output)
    # 추출된 특성 간의 평균제곱오차(MSE)를 계산
    mse_loss = F.mse_loss(gt_f, output_f)
    return mse_loss



#지정된 dir 하위에 있는 모든 오디오 파일을 재귀적으로 찾아 리스트로 반환
def glob_audio_files(dir):
    ext_list = ["wav", "mp3", "flac"]
    audio_files = []
    #확장자가 ext인 파일들을 찾음
    for ext in ext_list:
        audio_files.extend(glob.glob(
            os.path.join(dir, f"**/*.{ext}"), recursive=True))
    return audio_files
