import subprocess
import argparse
from pathlib import Path

# LLVC inference functions from the repository
from infer import load_model, load_audio, do_infer, save_audio


def extract_audio(video_path: Path, audio_path: Path) -> None:
    """Extracts audio from the input video using ffmpeg into a WAV file."""
    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-vn',  # no video
        '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian
        '-ar', '16000',          # 16 kHz
        '-ac', '1',              # mono
        str(audio_path)
    ]
    subprocess.run(cmd, check=True)


def modulate_audio(
    input_audio: Path,
    output_audio: Path,
    checkpoint_path: str,
    config_path: str,
    device: str = 'cpu'
) -> None:
    """
    Loads the LLVC model, runs inference on the extracted WAV, and saves the converted audio.

    Uses:
      - load_model(checkpoint_path, config_path)
      - load_audio    (path, sample_rate)
      - do_infer      (model, audio_tensor, chunk_factor, sr, stream=False)
      - save_audio    (tensor, out_path, sr)
    """
    # Load the model and sampling rate
    model, sr = load_model(checkpoint_path, config_path)
    # Load audio tensor
    audio_tensor = load_audio(str(input_audio), sr)
    # Convert (non-streaming)
    converted_tensor, _, _ = do_infer(model, audio_tensor, chunk_factor=1, sr=sr, stream=False)
    # Save the converted audio
    save_audio(converted_tensor.unsqueeze(0), str(output_audio), sr)


def merge_audio_video(
    original_video: Path,
    modulated_audio: Path,
    output_video: Path
) -> None:
    """Merges the modulated WAV back into the original video using ffmpeg."""
    cmd = [
        'ffmpeg', '-y',
        '-i', str(original_video),
        '-i', str(modulated_audio),
        '-c:v', 'copy',      # copy video stream
        '-map', '0:v:0',     # from first input
        '-map', '1:a:0',     # audio from second input
        '-c:a', 'aac',       # encode audio to AAC
        str(output_video)
    ]
    subprocess.run(cmd, check=True)


def convert_video_voice(args: argparse.Namespace) -> None:
    video_in = Path(args.input)
    # intermediate WAV files in same folder
    audio_orig = video_in.with_suffix('.wav')
    audio_mod = video_in.with_name(f"{video_in.stem}_modulated.wav")
    video_out = Path(args.output)

    print(f"[*] Extracting audio from {video_in}...")
    extract_audio(video_in, audio_orig)

    print(f"[*] Converting voice with LLVC model...")
    modulate_audio(
        audio_orig,
        audio_mod,
        args.checkpoint,
        args.config,
        args.device
    )

    print(f"[*] Merging converted audio into {video_out}...")
    merge_audio_video(video_in, audio_mod, video_out)

    print(f"[+] Conversion complete: {video_out.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Convert video voice using LLVC model inference'
    )
    parser.add_argument(
        '-i', '--input',    required=True,
        help='Path to input video (e.g., input.mp4)'
    )
    parser.add_argument(
        '-o', '--output',   required=True,
        help='Path to output video (e.g., output.mp4)'
    )
    parser.add_argument(
        '-c', '--checkpoint', required=True,
        help='LLVC model checkpoint path (.pth)'
    )
    parser.add_argument(
        '--config',         required=True,
        help='Path to LLVC config JSON (e.g., experiments/llvc/config.json)'
    )
    parser.add_argument(
        '-d', '--device',    default='cpu',
        choices=['cpu','cuda'],
        help='Device to run inference on'
    )
    args = parser.parse_args()
    convert_video_voice(args)


if __name__ == '__main__':
    main()
