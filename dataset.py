"""
Torch dataset object for synthetically rendered spatial data.
"""

import os # 운영체제(OS)와 상호작용할 수 있도록 도와주는 기본 라이브러리, 파일 경로나 디렉토리 관리, 환경 변수 접근 등을 할 때 사용
import torch # PyTorch 라이브러리를 가져오는 코드
from scipy.io.wavfile import read # WAV 파일을 읽기 위한 라이브러리
import os
import re
import glob # 파일 패턴 매칭을 위한 라이브러리
import librosa

def get_dataset(dir):
    original_files = glob.glob(os.path.join(dir, "*_original.wav"))
    converted_files = []
    for original_file in original_files:
        base_name = os.path.basename(original_file).replace("_original.wav", "")
        # 캐릭터 이름이 들어간 converted 파일 탐색
        match = glob.glob(os.path.join(dir, f"{base_name}_*converted.wav"))
        if match:
            converted_files.append(match[0])
        else:
            print(f"❌ 변환된 파일 없음: {base_name}")
            converted_files.append(None)  # 혹시 몰라서 None 넣음
    return original_files, converted_files

def load_wav(full_path, target_sr):
    # WAV 파일을 로드할 때 샘플링 레이트 맞추기
    data, sr = librosa.load(full_path, sr=target_sr)  # sr=target_sr로 지정하여 자동 리샘플링
    return data, target_sr  # 리샘플링된 데이터와 target_sr 반환

# 타겟 인덱싱
def get_target_index_from_filename(filename):
    character_map = {
        "conan": 0,
        "keroro": 1,
        "shinchan": 2
    }

    filename = filename.lower()  # 대소문자 무시
    for character, idx in character_map.items():
        if character in filename:
            return idx
    return -1  # 매칭 안 되는 경우


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

        original_data, o_sr = load_wav(original_wav, self.sr) # 원본 음성 데이터 및 샘플링 레이트 로드
        converted_data, c_sr = load_wav(converted_wav, self.sr) # 변환된 음성 데이터 및 샘플링 레이트 로드

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
        filename = os.path.basename(self.converted_files[idx])
        target_index = get_target_index_from_filename(filename)

        return converted, gt, target_index, filename


import os
from shutil import copyfile
import glob
import soundfile as sf


# 원본과 캐릭터 음성 경로
original_dir = "/content/drive/MyDrive/Colab_Notebooks/capstone/캡스톤_코난/aihub/02_wav/KsponSpeech_02/KsponSpeech_0125"
# 캐릭터별 변환 음성 폴더
converted_dirs = {
    "shinchan": "/content/drive/MyDrive/Colab_Notebooks/capstone/캡스톤_코난/짱구",
    "conan": "/content/drive/MyDrive/Colab_Notebooks/capstone/캡스톤_코난/코난/conan",
    "keroro": "/content/drive/MyDrive/Colab_Notebooks/capstone/캡스톤_코난/케로로"
}

# 출력 폴더
output_dir = "/content/drive/MyDrive/Colab_Notebooks/capstone/ModelAI/train"
os.makedirs(output_dir, exist_ok=True)

# original 3개 복사
original_files = glob.glob(os.path.join(original_dir, "*.wav"))[:3]
for i, orig in enumerate(original_files):
    dst = os.path.join(output_dir, f"speaker1_{i}_original.wav")
    copyfile(orig, dst)
    print(f"original {i} 복사 완료")

# 각 캐릭터에서 3개씩 복사 (keroro는 mp3 → wav 변환 필요)
for character, char_dir in converted_dirs.items():
    files = glob.glob(os.path.join(char_dir, "*"))
    selected_files = sorted([f for f in files if f.endswith((".wav", ".mp3"))])[:3]

    for j, src in enumerate(selected_files):
        dst = os.path.join(output_dir, f"speaker1_{j}_{character}_converted.wav")
        
        if src.endswith(".mp3"):
            print(f"{character} {j} mp3 → wav 변환 중")
            audio, sr = librosa.load(src, sr=16000) # MP3 파일 디코딩 (→ Raw PCM 오디오)
            sf.write(dst, audio, sr) # WAV 포맷으로 다시 인코딩해서 저장
        else:
            copyfile(src, dst)
            print(f"{character} {j} wav 복사 완료")

print("모든 캐릭터 음성 복사 완료")

import os
import glob
from collections import defaultdict

# 캐릭터 → 인덱스 매핑
character_to_index = {
    "conan": 0,
    "keroro": 1,
    "shinchan": 2
}

def get_parallel_dataset_by_index(dir):
    files = glob.glob(os.path.join(dir, "*.wav"))
    grouped = defaultdict(dict)  # {0: {'original': ..., 'conan': ..., ...}, 1: {...}, ...}
    result = []

    for f in files:
        filename = os.path.basename(f)

        # 예: speaker1_2_original.wav → prefix=speaker1, idx=2, type=original
        parts = filename.split("_")
        if len(parts) < 3:
            continue  # 예외 처리

        idx = int(parts[1])  # speaker1_2 → 2

        if "original" in filename:
            grouped[idx]["original"] = f
        else:
            for char in character_to_index:
                if char in filename:
                    grouped[idx][char] = f

    # 이제 같은 인덱스끼리 병렬 처리
    for idx, data in grouped.items():
        orig_path = data.get("original", None)
        if not orig_path:
            continue

        for char, index in character_to_index.items():
            converted_path = data.get(char, None)
            if converted_path:
                result.append((converted_path, orig_path, index))
                print(f"인덱스 {idx}: {char} (Index {index}) 병렬처리 완료")
            else:
                print(f"인덱스 {idx}: {char} 변환 파일 없음")

    return result

parallel_data = get_parallel_dataset_by_index(output_dir)

# 튜플 리스트 형태로 깔끔하게 출력
print("\n병렬 처리된 튜플 목록:")
for item in parallel_data:
    print(item)
