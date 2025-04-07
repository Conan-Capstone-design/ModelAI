"""
Torch dataset object for synthetically rendered spatial data.
"""

import os # 운영체제(OS)와 상호작용할 수 있도록 도와주는 기본 라이브러리, 파일 경로나 디렉토리 관리, 환경 변수 접근 등을 할 때 사용
import torch # PyTorch 라이브러리를 가져오는 코드
from scipy.io.wavfile import read # WAV 파일을 읽기 위한 라이브러리
import os
import re
import glob # 파일 패턴 매칭을 위한 라이브러리

def get_dataset(dir): # 주어진 디렉토리에서 '_original.wav' 파일과 대응하는 '_converted.wav' 파일을 찾는 함수
    original_files = glob.glob(os.path.join(dir, "*_original.wav")) # '_original.wav' 파일 목록을 가져옴
    converted_files = [] # 변환된 파일 목록을 저장할 리스트
    for original_file in original_files:
        converted_file = original_file.replace( 
            "_original.wav", "_converted.wav") # "_original.wav" -> '_converted.wav'로 변경
        converted_files.append(converted_file) # 변환된 파일 리스트에 저장
    return original_files, converted_files # 원본 파일과 변환된 파일 목록 반환


def load_wav(full_path): # 주어진 경로의 WAV 파일을 로드하는 함수
    sampling_rate, data = read(full_path) # WAV 파일의 샘플링 레이트와 데이터를 읽음
    return data, sampling_rate # 데이터와 샘플링 레이트 반환


class LLVCDataset(torch.utils.data.Dataset): # 음성 데이터를 로드하는 PyTorch Dataset 클래스

    def __init__(
        self,
        dir, # 데이터셋이 저장된 디렉토리
        sr, # 샘플링 레이트
        wav_len, # WAV 데이터 길이 (프레임 단위)
        dset # 데이터셋 유형 ('train', 'val', 'dev')
    ):
        assert dset in [ # 데이터셋 유형 ("train","val","dev")
            "train",
            "val",
            "dev"
        ], "`dset` must be one of ['train', 'val', 'dev']"
        self.dset = dset # 데이터셋 유형 저장
        file_dir = os.path.join(dir, dset) # 데이터셋 경로 설정
        self.wav_len = wav_len # WAV 길이 저장
        self.sr = sr # 샘플링 레이트 저장
        self.original_files, self.converted_files = get_dataset( # 파일 목록 불러오기
            file_dir
        )

    def __len__(self):
        return len(self.original_files) # 데이터셋의 크기 반환 (원본 파일 개수)

    def __getitem__(self, idx): # 인덱스에 해당하는 데이터를 로드하여 반환하는 함수
        original_wav = self.original_files[idx] # 원본 WAV 파일 경로 가져오기
        converted_wav = self.converted_files[idx] # 변환된 WAV 파일 경로 가져오기

        original_data, o_sr = load_wav(original_wav) # 원본 음성 데이터 및 샘플링 레이트 로드
        converted_data, c_sr = load_wav(converted_wav) # 변환된 음성 데이터 및 샘플링 레이트 로드

        assert o_sr == self.sr, f"Expected {self.sr}Hz, got {o_sr}Hz for file {original_wav}" # original_wav 샘플링 레이트 확인
        assert c_sr == self.sr, f"Expected {self.sr}Hz, got {c_sr}Hz for file {converted_wav}" # converted_wav 샘플링 레이트 확인

        converted = torch.from_numpy(original_data)  # original_data NumPy 배열을 PyTorch Tensor로 변환
        gt = torch.from_numpy(converted_data)  # converted_data NumPy 배열을 PyTorch Tensor로 변환

        converted = converted.unsqueeze(0).to(torch.float) / 32768  # 16비트 PCM 정규화 및 차원 추가
        gt = gt.unsqueeze(0).to(torch.float) / 32768 # 16비트 PCM 정규화 및 차원 추가

        if gt.shape[-1] < self.wav_len: # gt 데이터 길이가 `wav_len`보다 짧으면 0으로 패딩
            gt = torch.cat(
                (gt, torch.zeros(1, self.wav_len - gt.shape[-1])), dim=1)
        else: # `wav_len`보다 길면 잘라내기
            gt = gt[:, : self.wav_len]

        if converted.shape[-1] < self.wav_len: # 변환된 데이터 길이가 `wav_len`보다 짧으면 0으로 패딩
            converted = torch.cat(
                (converted, torch.zeros(1, self.wav_len - converted.shape[-1])), dim=1
            )
        else: # `wav_len`보다 길면 잘라내기
            converted = converted[:, : self.wav_len]

        #임의로 파일 이름에 따라 타겟 인덱스 지정해줌
        filename = os.path.basename(self.original_files[idx])
        match = re.search(r"speaker(\d+)_", filename)
        target_index = int(match.group(1)) if match else 0

        return converted, gt, target_index
