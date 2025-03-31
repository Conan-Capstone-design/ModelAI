import math # 수학 연산을 위한 기본 라이브러리
import os # 운영체제와 상호작용할 수 있는 기능 제공 (파일 및 디렉토리 관련)
import random # 난수 생성 관련 라이브러리
import torch # PyTorch 라이브러리 (딥러닝, 텐서 연산)
from torch import nn # 신경망 구축을 위한 모듈 (예: 신경망 레이어 정의)
import torch.nn.functional as F # PyTorch의 함수형 API (예: 활성화 함수, 손실 함수 등)
import torch.utils.data # 데이터 로딩 및 배치 처리 모듈
import numpy as np # 배열 및 행렬 연산을 위한 라이브러리
import librosa # 오디오 신호 처리 라이브러리 (예: MFCC, Spectrogram 변환)
import librosa.util as librosa_util # librosa 유틸리티 함수 사용
from librosa.util import normalize, pad_center, tiny # librosa에서 직접 유틸리티 함수 임포트
from scipy.signal import get_window # 윈도우 함수 생성 (예: 해닝 윈도우)
from scipy.io.wavfile import read # WAV 파일 읽기
from librosa.filters import mel as librosa_mel_fn # Mel 필터 생성 함수

MAX_WAV_VALUE = 32768.0 # 최대 WAV 신호 값 (16-bit PCM에서 최대값)

# 동적 범위 압축을 수행하는 함수(로그스케일)
# x: 입력 텐서 (오디오 스펙트로그램 값)
# C: 압축 계수
# clip_val: 로그 변환을 위해 최소값을 설정 (0 이하의 값 방지)
def dynamic_range_compression_torch(x, C=1, clip_val=1e-5): 
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)

# 동적 범위 압축 해제 함수(지수 스케일)
# x: 압축된 입력 텐서
# C: 압축 시 사용한 계수
def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C

# 스펙트럼 정규화 (로그 스케일로 변환)
def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

# 스펙트럼 정규화 원래대로 되돌리는 함수
def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

# Mel 필터와 해닝 창 저장용 딕셔너리
mel_basis = {}
hann_window = {}

# 오디오 -> 스펙트로그램 생성 함수
def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
# y: 입력 오디오 텐서
# n_fft: FFT 크기 (프레임 크기)
# sampling_rate: 샘플링 레이트 (Hz)
# hop_size: 프레임 간 간격
# win_size: 윈도우 크기
# center: 중앙 정렬 여부

# 오디오 데이터의 최소/최대 값이 -1~1을 초과하는지 확인
    if torch.min(y) < -1.: # 최소값 -1 미만일 때 경고
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.: # 최대값 1 초과일 때 경고
        print('max value is ', torch.max(y))

    global hann_window # 전역 변수 사용
    dtype_device = str(y.dtype) + '_' + str(y.device) # 데이터 타입과 장치 정보 저장
    wnsize_dtype_device = str(win_size) + '_' + dtype_device # 윈도우 크기와 장치 정보 저장
    # 해닝 윈도우가 존재하지 않으면 생성 후 저장
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(
            win_size).to(dtype=y.dtype, device=y.device)
    # 패딩 추가 (경계 부분 처리)
    y = torch.nn.functional.pad(y.unsqueeze(
        1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    # STFT 변환 수행 (단시간 푸리에 변환)
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    # 복소수를 실수 형태로 변환
    spec = torch.view_as_real(spec)
    # 복소수 크기 계산 (제곱 후 제곱근)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec

# 스펙트로그램 -> Mel 스펙트로그램으로 변환하는 함수
def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    dtype_device = str(spec.dtype) + '_' + str(spec.device) # 데이터 타입과 장치 정보 저장
    fmax_dtype_device = str(fmax) + '_' + dtype_device # 최대 주파수 정보 저장
    # Mel 필터가 존재하지 않으면 생성 후 저장
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft,
                             n_mels=num_mels, fmin=fmin, fmax=fmax) # 멜 필터 생성
        mel_basis[fmax_dtype_device] = torch.from_numpy(
            mel).to(dtype=spec.dtype, device=spec.device) # Torch Tensor로 변환 후 캐시에 저장
    # Mel 필터 적용
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    # 정규화
    spec = spectral_normalize_torch(spec)
    return spec

#  오디오 -> 멜 스펙트로그램을 생성하는 함수
def mel_spectrogram_torch(y, sampling_rate, n_fft, num_mels, hop_size, win_size, fmin, fmax, center=False):
# y (torch.Tensor): 오디오 신호 (1D Tensor)
# sampling_rate (int): 샘플링 레이트 (Hz)
# n_fft (int): FFT 크기 (프레임 크기)
# num_mels (int): 생성할 멜 필터 개수
# hop_size (int): STFT 프레임 간격
# win_size (int): 윈도우 크기
# fmin (float): 멜 필터의 최소 주파수
# fmax (float): 멜 필터의 최대 주파수
# center (bool): 입력 신호를 패딩하여 중앙 정렬할지 여부

    # 오디오 데이터의 최소/최대 값이 -1~1을 초과하는지 확인
    # 입력 오디오 신호의 최소값이 -1보다 작은 경우 경고 출력
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    # 입력 오디오 신호의 최대값이 1보다 큰 경우 경고 출력
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    # 전역 변수 선언 (멜 필터와 해닝 윈도우 저장용)
    global mel_basis, hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device) # 데이터 타입과 장치 정보 (ex: "float32_cuda:0")를 문자열로 저장
    fmax_dtype_device = str(fmax) + '_' + dtype_device # fmax와 dtype_device를 조합한 키 생성 (멜 필터 캐시를 위한 키)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device # 윈도우 크기와 dtype_device를 조합한 키 생성 (해닝 윈도우 캐시를 위한 키)
    # Mel 필터가 존재하지 않으면 생성 후 저장
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft,
                             n_mels=num_mels, fmin=fmin, fmax=fmax) # 멜 필터 생성
        mel_basis[fmax_dtype_device] = torch.from_numpy(
            mel).to(dtype=y.dtype, device=y.device) # Torch Tensor로 변환 후 캐시에 저장
    # 해닝 윈도우가 존재하지 않으면 생성 후 저장
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(
            win_size).to(dtype=y.dtype, device=y.device)

    # 패딩 적용(STFT 입력 크기 맞추기)
    y = torch.nn.functional.pad(y.unsqueeze( # (batch, 1, time) 형태로 변경
        1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), # 앞뒤로 패딩 추가
        mode='reflect') # 반사 패딩 적용 (끝값을 반사하여 채움)
    y = y.squeeze(1) # (batch, time) 형태로 되돌림

    # STFT 변환 수행 및 Mel 변환 적용
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    # 복소수 형태의 스펙트로그램을 실수 형태로 변환
    spec = torch.view_as_real(spec)
    # STFT 결과를 진폭 스펙트로그램으로 변환
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    # Mel 변환 적용 (STFT -> Mel 스펙트로그램 변환)
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    # 로그 스케일 정규화 (Dynamic Range Compression)
    spec = spectral_normalize_torch(spec)

    # 최종 멜 스펙트로그램
    return spec
