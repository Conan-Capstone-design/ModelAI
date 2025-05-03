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
    model = Net(**config['model_params'])  # 설정 파일 기반으로 모델 초기화
    model.load_state_dict(torch.load(
        checkpoint_path, map_location="cpu")['model'])  # 체크포인트에서 가중치 불러오기
    return model, config['data']['sr']  # 모델과 샘플링 레이트 반환


def load_audio(audio_path, sample_rate):
    """ 오디오 파일을 로드하고 원하는 샘플링 레이트로 변환하는 함수
    
    Args:
        audio_path (str): 로드할 오디오 파일 경로
        sample_rate (int): 변환할 샘플링 레이트
    
    Returns:
        torch.Tensor: 변환된 오디오 데이터
    """
    audio, sr = torchaudio.load(audio_path)  # 오디오 파일 로드
    audio = audio.mean(0, keepdim=False)  # 스테레오 오디오를 모노로 변환 (채널 평균)
    audio = torchaudio.transforms.Resample(sr, sample_rate)(audio)  # 샘플링 레이트 변환
    return audio


def save_audio(audio, audio_path, sample_rate):
    """ 변환된 오디오 데이터를 파일로 저장하는 함수
    
    Args:
        audio (torch.Tensor): 저장할 오디오 데이터
        audio_path (str): 저장할 파일 경로
        sample_rate (int): 샘플링 레이트
    """
    torchaudio.save(audio_path, audio, sample_rate)


def infer(model, audio, target_index):
    return model(audio.unsqueeze(0).unsqueeze(0), target_index=target_index).squeeze(0)



def infer_stream(model, audio, chunk_factor, sr):
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
    L = model.L  # 모델의 L 파라미터
    chunk_len = model.dec_chunk_size * L * chunk_factor  # 청크 크기 계산
    
    original_len = len(audio)
    if len(audio) % chunk_len != 0:
        pad_len = chunk_len - (len(audio) % chunk_len)
        audio = torch.nn.functional.pad(audio, (0, pad_len))  # 패딩 추가하여 크기 정렬
    
    audio = torch.cat((audio[L:], torch.zeros(L)))  # 오디오를 L만큼 이동하여 앞 부분 컨텍스트 확보
    audio_chunks = torch.split(audio, chunk_len)  # 오디오를 청크로 분할
    
    # 각 청크 앞에 컨텍스트 추가 (이전 청크의 마지막 L * 2 샘플 사용)
    new_audio_chunks = []
    for i, a in enumerate(audio_chunks):
        front_ctx = audio_chunks[i - 1][-L * 2:] if i > 0 else torch.zeros(L * 2)
        new_audio_chunks.append(torch.cat([front_ctx, a]))
    audio_chunks = new_audio_chunks
    
    outputs = []
    times = []
    with torch.inference_mode():
        enc_buf, dec_buf, out_buf = model.init_buffers(1, torch.device('cpu'))  # 버퍼 초기화
        convnet_pre_ctx = model.convnet_pre.init_ctx_buf(1, torch.device('cpu')) if hasattr(model, 'convnet_pre') else None
        
        for chunk in audio_chunks:
            start = time.time()
            output, enc_buf, dec_buf, out_buf, convnet_pre_ctx = model(
                chunk.unsqueeze(0).unsqueeze(0), enc_buf, dec_buf, out_buf,
                convnet_pre_ctx, pad=(not model.lookahead)
            )
            outputs.append(output)
            times.append(time.time() - start)
    
    outputs = torch.cat(outputs, dim=2)
    avg_time = np.mean(times)
    rtf = (chunk_len / sr) / avg_time  # 실시간 변환 비율(RTF) 계산
    e2e_latency = ((2 * L + chunk_len) / sr + avg_time) * 1000  # 전체 지연 시간(ms) 계산
    
    outputs = outputs[:, :, :original_len].squeeze(0)  # 패딩 제거
    return outputs, rtf, e2e_latency


def do_infer(model, audio, chunk_factor, sr, stream, target_index):
    with torch.no_grad():
        if stream:
            return infer_stream(model, audio, chunk_factor, sr)
        else:
            return infer(model, audio, target_index), None, None


def main():
    """ 명령줄 인자를 받아 오디오 변조를 실행하는 메인 함수 """
    parser = argparse.ArgumentParser()
    # parser.add_argument('--checkpoint_path', '-p', type=str, default='llvc_models/models/checkpoints/llvc/G_500000.pth', help='모델 체크포인트 경로')
    # parser.add_argument('--config_path', '-c', type=str, default='experiments/llvc/config.json', help='모델 설정 파일 경로')
    parser.add_argument('--checkpoint_path', '-p', type=str, default='/content/drive/MyDrive/project/ModelAI/llvc_nc/checkpoints/G_300.pth', help='모델 체크포인트 경로')
    parser.add_argument('--config_path', '-c', type=str, default='/content/drive/MyDrive/project/ModelAI/llvc_nc/config.json', help='모델 설정 파일 경로')
    parser.add_argument('--fname', '-f', type=str, default='/content/drive/MyDrive/캡스톤_코난/test_wavs',help='오디오 파일 또는 디렉토리 경로')
    parser.add_argument('--out_dir', '-o', type=str, default='converted_out', help='출력 오디오 저장 디렉토리')
    parser.add_argument('--chunk_factor', '-n', type=int, default=1, help='청크 인자 (스트리밍 처리용)')
    parser.add_argument('--stream', '-s', action='store_true', help='스트리밍 인퍼런스 사용 여부')
    parser.add_argument('--target_index', '-t', type=int, default=0, help='타겟 화자 인덱스 (예: 0=코난, 1=케로로, 2=짱구)')
    args = parser.parse_args()
    
    model, sr = load_model(args.checkpoint_path, args.config_path)
    os.makedirs(args.out_dir, exist_ok=True)
    
    if os.path.isdir(args.fname):
        fnames = glob_audio_files(args.fname)
        for fname in tqdm(fnames):
            audio = load_audio(fname, sr)
            out, _, _ = do_infer(model, audio, args.chunk_factor, sr, args.stream, torch.tensor([args.target_index]))
            save_audio(out, os.path.join(args.out_dir, os.path.basename(fname)), sr)
        print(f"Saved outputs to {args.out_dir}")
    
if __name__ == '__main__':
    main()