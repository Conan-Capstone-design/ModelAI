"""
Torch dataset object for synthetically rendered spatial data.
"""

import os # 운영체제(OS)와 상호작용할 수 있도록 도와주는 기본 라이브러리, 파일 경로나 디렉토리 관리, 환경 변수 접근 등을 할 때 사용
import torch # PyTorch 라이브러리를 가져오는 코드
from scipy.io.wavfile import read # WAV 파일을 읽기 위한 라이브러리
import re
import glob # 파일 패턴 매칭을 위한 라이브러리
from collections import defaultdict


def get_dataset_conan_only(dir):
    all_wavs = glob.glob(os.path.join(dir, "**", "*.wav"), recursive=True)
    grouped = defaultdict(dict)
    result = []

    for f in all_wavs:
        fname = os.path.basename(f)

        if fname.endswith("_conan_converted.wav"):
            prefix = fname.replace("_conan_converted.wav", "")
            grouped[prefix]["conan"] = f
        elif "_converted" not in fname:
            prefix = fname.replace(".wav", "")
            grouped[prefix]["original"] = f

    for prefix, file_dict in grouped.items():
        orig = file_dict.get("original")
        conan = file_dict.get("conan")
        if orig and conan:
            result.append((orig, conan, 0))
        else:
            print(f"[스킵] {prefix} → conan 변환 파일이 없거나 original 없음")

    return result


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
        # self.original_files, self.converted_files = get_dataset( # 파일 목록 불러오기
        #     file_dir
        # )
        self.data = get_dataset_conan_only(file_dir)
        
    def __len__(self):
        return len(self.data)  # ← 이게 맞음!

    def __getitem__(self, idx):
        original_wav, converted_wav, target_index = self.data[idx]

        original_data, o_sr = load_wav(original_wav)
        converted_data, c_sr = load_wav(converted_wav)

        assert o_sr == self.sr, f"Expected {self.sr}Hz, got {o_sr}Hz for file {original_wav}"
        assert c_sr == self.sr, f"Expected {self.sr}Hz, got {c_sr}Hz for file {converted_wav}"

        converted = torch.from_numpy(original_data).unsqueeze(0).float() / 32768
        gt = torch.from_numpy(converted_data).unsqueeze(0).float() / 32768

        if gt.shape[-1] < self.wav_len:
            gt = torch.cat((gt, torch.zeros(1, self.wav_len - gt.shape[-1])), dim=1)
        else:
            gt = gt[:, : self.wav_len]

        if converted.shape[-1] < self.wav_len:
            converted = torch.cat((converted, torch.zeros(1, self.wav_len - converted.shape[-1])), dim=1)
        else:
            converted = converted[:, : self.wav_len]

        return converted, gt, target_index