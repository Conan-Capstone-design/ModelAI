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
            print(f"변환된 파일 없음: {base_name}")
            converted_files.append(None)  # 혹시 몰라서 None 넣음
    return original_files, converted_files

def get_dataset_any_to_many(dir):
    original_files = glob.glob(os.path.join(dir, "*_original.wav"))
    data = []

    character_map = {
        "conan": 0,
        "keroro": 1,
        "shinchan": 2
    }

    for orig_path in original_files:
        base = os.path.basename(orig_path).replace("_original.wav", "")
        for character, index in character_map.items():
            conv_path = os.path.join(dir, f"{base}_{character}_converted.wav")
            if os.path.exists(conv_path):
                data.append((orig_path, conv_path, index))
            else:
                print(f"변환 파일 없음: {conv_path}")

    return data

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
        self.data = get_dataset_any_to_many(file_dir)
        # self.original_files, self.converted_files = get_dataset( # 파일 목록 불러오기
        #     file_dir
        # )

    def __len__(self):
        return len(self.data) # 데이터셋의 크기 반환 (원본 파일 개수)
    
    def get_target_name(self):
        return {0: "conan", 1: "keroro", 2: "shinchan"}[self.target_index]

    def __getitem__(self, idx):
        original_path, converted_path, target_index = self.data[idx]

        original_data, o_sr = load_wav(original_path, self.sr)
        converted_data, c_sr = load_wav(converted_path, self.sr)

        assert o_sr == self.sr
        assert c_sr == self.sr

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
# import json

# # config.json 불러오기
# with open("llvc_nc/config.json", "r") as f:
#     config = json.load(f)

# # config에서 dir 값 가져오기
# base_dir = config["data"]["dir"]

# import os
# from shutil import copyfile
# import glob
# import soundfile as sf

# # 원본과 캐릭터 음성 경로
# original_dir = os.path.join(base_dir, "aihub/02_wav/KsponSpeech_02/KsponSpeech_0125")
# # 캐릭터별 변환 음성 폴더
# converted_dirs = {
#     "shinchan": os.path.join(base_dir, "짱구"),
#     "conan": os.path.join(base_dir, "코난/conan"),
#     "keroro": os.path.join(base_dir, "케로로")
# }

# # 출력 폴더
# output_dir = os.path.join(base_dir, "train")
# os.makedirs(output_dir, exist_ok=True)

# # original 3개 복사
# original_files = glob.glob(os.path.join(original_dir, "*.wav"))[:3]
# for i, orig in enumerate(original_files):
#     dst = os.path.join(output_dir, f"speaker1_{i}_original.wav")
#     copyfile(orig, dst)
#     print(f"original {i} 복사 완료")

# # 각 캐릭터에서 3개씩 복사 (keroro는 mp3 → wav 변환 필요)
# for character, char_dir in converted_dirs.items():
#     files = glob.glob(os.path.join(char_dir, "*"))
#     selected_files = sorted([f for f in files if f.endswith((".wav", ".mp3"))])[:3]

#     for j, src in enumerate(selected_files):
#         dst = os.path.join(output_dir, f"speaker1_{j}_{character}_converted.wav")
        
#         if src.endswith(".mp3"):
#             print(f"{character} {j} mp3 → wav 변환 중")
#             audio, sr = librosa.load(src, sr=16000) # MP3 파일 디코딩 (→ Raw PCM 오디오)
#             sf.write(dst, audio, sr) # WAV 포맷으로 다시 인코딩해서 저장
#         else:
#             copyfile(src, dst)
#             print(f"{character} {j} wav 복사 완료")

# print("모든 캐릭터 음성 복사 완료")

# import os
# import glob
# from collections import defaultdict

# # 캐릭터 → 인덱스 매핑
# character_to_index = {
#     "conan": 0,
#     "keroro": 1,
#     "shinchan": 2
# }

# def get_parallel_dataset_by_index(dir):
#     files = glob.glob(os.path.join(dir, "*.wav"))
#     grouped = defaultdict(dict)  # {0: {'original': ..., 'conan': ..., ...}, 1: {...}, ...}
#     result = []

#     for f in files:
#         filename = os.path.basename(f)

#         # 예: speaker1_2_original.wav → prefix=speaker1, idx=2, type=original
#         parts = filename.split("_")
#         if len(parts) < 3:
#             continue  # 예외 처리

#         idx = int(parts[1])  # speaker1_2 → 2

#         if "original" in filename:
#             grouped[idx]["original"] = f
#         else:
#             for char in character_to_index:
#                 if char in filename:
#                     grouped[idx][char] = f

#     # 같은 인덱스끼리 병렬 처리
#     for idx, data in grouped.items():
#         orig_path = data.get("original", None)
#         if not orig_path:
#             continue

#         for char, index in character_to_index.items():
#             converted_path = data.get(char, None)
#             if converted_path:
#                 result.append((converted_path, orig_path, index))
#                 print(f"인덱스 {idx}: {char} (Index {index}) 병렬처리 완료")
#             else:
#                 print(f"인덱스 {idx}: {char} 변환 파일 없음")

#     return result

# parallel_data = get_parallel_dataset_by_index(output_dir)

# # 튜플 리스트 형태로 깔끔하게 출력
# print("\n병렬 처리된 튜플 목록:")
# for item in parallel_data:
#     print(item)
