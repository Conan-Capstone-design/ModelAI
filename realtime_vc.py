import torch
import numpy as np
import sounddevice as sd
from infer import load_model
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, required=True)
parser.add_argument('--config_path', type=str, required=True)
parser.add_argument('--input_device', type=int, default=1)
parser.add_argument('--output_device', type=int, default=4)
parser.add_argument('--target_index', type=int, required=True)
args = parser.parse_args()

# ───── 설정 ─────
CHECKPOINT_PATH = args.checkpoint_path
CONFIG_PATH = args.config_path
SR = 16000
CHUNK_DURATION = 1.0
CHUNK_SIZE = int(SR * CHUNK_DURATION)

input_device = args.input_device
output_device = args.output_device
TARGET_INDEX = torch.tensor([args.target_index])

# 모델 로딩
print("모델 로딩 중...")
model, sr = load_model(CHECKPOINT_PATH, CONFIG_PATH)
model.eval()
device = torch.device("cpu")

# 버퍼 초기화
enc_buf, dec_buf, out_buf = model.init_buffers(1, device)
convnet_pre_ctx = model.convnet_pre.init_ctx_buf(1, device) if hasattr(model, 'convnet_pre') else None

print("모델 로딩 완료")

# 시각화용 출력 버퍼
output_buffer = []

# ───── 콜백 함수 ─────
# ───── 콜백 함수 ─────
def callback(indata, outdata, frames, time, status):
    global enc_buf, dec_buf, out_buf, convnet_pre_ctx, output_buffer
    print(f"[CALLBACK] frames={frames}, indata_mean={indata.mean():.4f}")
    if status:
        print("Stream status:", status)

    # indata: (frames x 1) NumPy array
    audio_tensor = torch.FloatTensor(indata[:, 0])              # shape=(frames,)
    audio_tensor = torch.cat([audio_tensor, torch.zeros(model.L)])
    audio_tensor[torch.abs(audio_tensor) < 0.01] = 0

    with torch.no_grad():
        output, enc_buf, dec_buf, out_buf, convnet_pre_ctx = model(
            x               = audio_tensor.unsqueeze(0).unsqueeze(0),  # (1 x 1 x T)
            target_index    = TARGET_INDEX,                             # torch.tensor([0])
            init_enc_buf    = enc_buf,
            init_dec_buf    = dec_buf,
            init_out_buf    = out_buf,
            convnet_pre_ctx = convnet_pre_ctx,
            pad             = not model.lookahead
        )

    # 모델이 리턴한 텐서 (1 x 1 x T_out) 형태 → 먼저 1차원으로 펴기
    out_tensor = output[0, 0]                     # shape: (T_out,)
    # 필요한 만큼만(frames 샘플) 잘라냄
    out_samples = out_tensor[:frames]             # shape: (min(T_out, frames),)
    out_np = torch.tanh(out_samples).cpu().numpy()  # NumPy array, 길이 N ≤ frames

    # outdata: (frames x 1) NumPy 배열 → 일단 무음(0)으로 초기화
    outdata[:] = 0.0
    # out_np가 차지하는 구간만 덮어씀 (나머지는 0으로 남음)
    outdata[0 : out_np.shape[0], 0] = out_np.reshape(-1)

    # (선택) 시각화를 위해 버퍼에 저장해 두고 싶다면:
    # output_buffer.append(out_np.copy())

    
# ───── 스트리밍 시작 ─────
print("실시간 음성 변조 시작 (Ctrl+C로 종료)")
try:
    with sd.Stream(
        samplerate=SR,
        blocksize=CHUNK_SIZE,
        dtype='float32',
        channels=1,
        callback=callback,
        device=(input_device, output_device)
    ):
        while True:
            sd.sleep(1000)
except KeyboardInterrupt:
    print("스트리밍 종료됨")
