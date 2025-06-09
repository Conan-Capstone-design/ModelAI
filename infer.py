from model import Net
import torch
import torchaudio
import time
import numpy as np
import argparse
import json
import os
from utils import glob_audio_files
from tqdm import tqdm


def load_model(checkpoint_path, config_path):
    """ 모델을 로드하고 체크포인트를 불러오는 함수
    
    Args:
        checkpoint_path (str): 저장된 모델 가중치 파일 경로
        config_path (str): 모델 설정 파일 경로
    
    Returns:
        model (Net): 로드된 모델
        int: 모델이 사용할 샘플링 레이트
    """
    with open(config_path) as f:
        config = json.load(f)
    model = Net(**config['model_params'])
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, config['data']['sr']


def load_audio(audio_path, sample_rate):
    """ 오디오 파일을 로드하고 원하는 샘플링 레이트로 변환하는 함수
    
    Args:
        audio_path (str): 로드할 오디오 파일 경로
        sample_rate (int): 변환할 샘플링 레이트
    
    Returns:
        torch.Tensor: 변환된 오디오 데이터
    """
    audio, sr = torchaudio.load(audio_path)
    audio = audio.mean(0)  # (채널 평균) → 1차원 Tensor
    if sr != sample_rate:
        audio = torchaudio.transforms.Resample(sr, sample_rate)(audio)
    return audio


def save_audio(audio, audio_path, sample_rate):
    """ 변환된 오디오 데이터를 파일로 저장하는 함수
    
    Args:
        audio (torch.Tensor): 저장할 오디오 데이터
        audio_path (str): 저장할 파일 경로
        sample_rate (int): 샘플링 레이트
    """
    # audio는 1D Tensor 또는 (1 x T) Tensor
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    torchaudio.save(audio_path, audio, sample_rate)


def infer(model, audio: torch.Tensor, target_index: torch.Tensor):
    """
    모델을 한 번에(in-batch) 돌리는 함수
    audio: 1차원 Tensor (T)
    target_index: torch.Tensor([k]) 모양
    """
    with torch.no_grad():
        x = audio.unsqueeze(0).unsqueeze(0)  # (1 x 1 x T)
        out = model(x, target_index=target_index)  # (1 x 1 x T)
        return out.squeeze(0).squeeze(0)  # 1D Tensor


def infer_stream(model, audio: torch.Tensor, target_index: torch.Tensor, chunk_factor: int, sr: int):
    """ 스트리밍 방식으로 오디오를 처리하는 함수
    
    Args:
        model (Net): 변조할 모델
        audio (torch.Tensor): 입력 오디오 데이터
        chunk_factor (int): 스트리밍 처리 시 청크 크기 조절 계수
        sr (int): 샘플링 레이트
    
    Returns:
        torch.Tensor: 변조된 오디오 데이터
        float: 실시간 변환 비율(RTF)
        float: 엔드 투 엔드 지연 시간(ms)
    """
    L = model.L
    chunk_len = model.dec_chunk_size * L * chunk_factor

    original_len = len(audio)
    if len(audio) % chunk_len != 0:
        pad_len = chunk_len - (len(audio) % chunk_len)
        audio = torch.nn.functional.pad(audio, (0, pad_len))

    # 앞쪽 L 샘플을 미리 잘라서 뒤로 붙여놓기
    audio = torch.cat((audio[L:], torch.zeros(L)))
    audio_chunks = torch.split(audio, chunk_len)

    # 前 청크의 끝 L*2 샘플을 현재 청크 앞에 붙여서 컨텍스트 확보
    new_audio_chunks = []
    for i, a in enumerate(audio_chunks):
        if i > 0:
            front_ctx = audio_chunks[i - 1][-L * 2:]
        else:
            front_ctx = torch.zeros(L * 2)
        new_audio_chunks.append(torch.cat([front_ctx, a]))
    audio_chunks = new_audio_chunks

    outputs = []
    times = []
    with torch.inference_mode():
        # 버퍼 초기화 (batch_size=1, device=cpu)
        enc_buf, dec_buf, out_buf = model.init_buffers(1, torch.device('cpu'))
        convnet_pre_ctx = None
        if hasattr(model, 'convnet_pre'):
            convnet_pre_ctx = model.convnet_pre.init_ctx_buf(1, torch.device('cpu'))

        for chunk in audio_chunks:
            start = time.time()

            # chunk 텐서를 (1 x 1 x T_chunk) 형태로 만들어서 모델에 넘겨줌
            output, enc_buf, dec_buf, out_buf, convnet_pre_ctx = model(
                x=chunk.unsqueeze(0).unsqueeze(0),
                target_index=target_index,
                init_enc_buf=enc_buf,
                init_dec_buf=dec_buf,
                init_out_buf=out_buf,
                convnet_pre_ctx=convnet_pre_ctx,
                pad=not model.lookahead
            )
            outputs.append(output)  # (1 x 1 x out_chunk_len)
            times.append(time.time() - start)

    # 청크별 출력을 시간축(dim=2) 기준으로 붙여서 하나로 합침
    outputs = torch.cat(outputs, dim=2)  # (1 x 1 x total_time)
    avg_time = np.mean(times)
    rtf = (chunk_len / sr) / avg_time
    e2e_latency = ((2 * L + chunk_len) / sr + avg_time) * 1000

    # 원래 오디오 길이만큼 잘라서 1D Tensor로 리턴
    outputs = outputs[:, :, :original_len].squeeze(0).squeeze(0)
    return outputs, rtf, e2e_latency


def do_infer(model, audio: torch.Tensor, chunk_factor: int, sr: int, stream: bool, target_index: torch.Tensor):
    if stream:
        return infer_stream(model, audio, target_index, chunk_factor, sr)
    else:
        return infer(model, audio, target_index), None, None


def main():
    """ 명령줄 인자를 받아 오디오 변조를 실행하는 메인 함수 """
    parser = argparse.ArgumentParser()
    # parser.add_argument('--checkpoint_path', '-p', type=str, default='llvc_models/models/checkpoints/llvc/G_500000.pth', help='모델 체크포인트 경로')
    # parser.add_argument('--config_path', '-c', type=str, default='experiments/llvc/config.json', help='모델 설정 파일 경로')
    parser.add_argument('--checkpoint_path', '-p', type=str, default='/content/ModelAI/llvc_nc/G_765000.pth')
    parser.add_argument('--config_path', '-c', type=str, default='/content/ModelAI/llvc_nc/config.json')
    parser.add_argument('--fname', '-f', type=str, default='test_wavs',
                        help='오디오 파일 또는 디렉토리 경로')
    parser.add_argument('--out_dir', '-o', type=str, default='converted_out')
    parser.add_argument('--chunk_factor', '-n', type=int, default=1,
                        help='스트리밍 모드일 때 청크 크기 배수')
    parser.add_argument('--stream', '-s', action='store_true',
                        help='스트리밍(infer_stream) 모드 사용 여부')
    parser.add_argument('--target_index', '-t', type=int, default=0,
                        help='타겟 화자 인덱스 (예: 0=코난, 1=케로로, 2=짱구)')
    args = parser.parse_args()

    model, sr = load_model(args.checkpoint_path, args.config_path)
    os.makedirs(args.out_dir, exist_ok=True)

    # target_index를 반드시 1차원 Tensor로 만들어서 모델에 넘겨줘야 함
    target_index_tensor = torch.tensor([args.target_index])

    # 단일 파일인지, 디렉토리인지 확인
    if os.path.isdir(args.fname):
        fnames = glob_audio_files(args.fname)
    else:
        fnames = [args.fname]

    for fname in tqdm(fnames):
        audio = load_audio(fname, sr)  # 1D Tensor
        out, rtf, latency = do_infer(
            model, audio, args.chunk_factor, sr, args.stream, target_index_tensor
        )
        save_audio(out, os.path.join(args.out_dir, os.path.basename(fname)), sr)
        if args.stream:
            print(f"[{os.path.basename(fname)}] RTF={rtf:.3f}, Latency={latency:.1f}ms")

    print(f"Saved outputs to `{args.out_dir}`")


if __name__ == '__main__':
    main()
