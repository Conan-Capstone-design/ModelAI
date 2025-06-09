import sounddevice as sd
import torch
import numpy as np
from model import Net
import json
import time

# ========= 설정 =========
CHECKPOINT_PATH = "./llvc_nc/checkpoints/G_300.pth"
CONFIG_PATH = "./llvc_nc/config.json"
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5  # 청크 길이 (초)
TARGET_INDEX = 0  # 변조할 타겟 화자 인덱스 (0=코난, 1=케로로, 2=짱구 등)

# ========= 모델 로드 =========
with open(CONFIG_PATH) as f:
    config = json.load(f)

model = Net(**config["model_params"])
state = torch.load(CHECKPOINT_PATH, map_location="cpu")
model.load_state_dict(state["model"])
model.eval()

L = model.L
chunk_size = model.dec_chunk_size * L * 1  # chunk_factor=1 기준
buffer_size = int(SAMPLE_RATE * CHUNK_DURATION)

print("실시간 음성 변조 시작 (Ctrl+C로 종료하세요)")

# ========= 초기 버퍼 상태 =========
enc_buf, dec_buf, out_buf = model.init_buffers(1, torch.device("cpu"))
convnet_pre_ctx = model.convnet_pre.init_ctx_buf(1, torch.device("cpu")) if hasattr(model, "convnet_pre") else None

# ========= 콜백 함수 =========
def callback(indata, outdata, frames, time_info, status):
    global enc_buf, dec_buf, out_buf, convnet_pre_ctx
    audio = torch.from_numpy(indata[:, 0]).float()

    # L-padding 및 배치 차원 추가
    audio = torch.cat([audio, torch.zeros(L)])
    with torch.inference_mode():
        output, enc_buf, dec_buf, out_buf, convnet_pre_ctx = model(
            audio.unsqueeze(0).unsqueeze(0),
            enc_buf,
            dec_buf,
            out_buf,
            convnet_pre_ctx,
            target_index=torch.tensor([TARGET_INDEX]),
            pad=not model.lookahead
        )
    # 출력 오디오 자르기 및 넘파이로 변환
    audio_out = output[0, 0, :frames].cpu().numpy()
    outdata[:] = np.expand_dims(audio_out, axis=1)

# ========= 스트림 시작 =========
with sd.Stream(
    samplerate=SAMPLE_RATE,
    blocksize=buffer_size,
    dtype="float32",
    channels=1,
    callback=callback
):
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("변조 종료됨")
