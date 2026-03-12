import concurrent.futures
import gc
import os
import re
import subprocess

from faster_whisper import WhisperModel

MODEL_SIZE = "large-v3"
SUPPORTED_EXTENSIONS = (
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
)


def format_timestamp(seconds):
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{int(s):02d}"


def get_sorted_media_files(input_dir):
    files = [
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith(SUPPORTED_EXTENSIONS) and not f.startswith(".temp_")
    ]

    def sort_key(filename):
        numbers = re.findall(r"\d+", filename)
        if numbers:
            return int(numbers[0])
        return filename.lower()

    return sorted(files, key=sort_key)


def prepare_media_file(input_dir, filename):
    input_path = os.path.join(input_dir, filename)
    base_name = os.path.splitext(filename)[0]
    temp_wav = os.path.join(input_dir, f".temp_{base_name}.wav")
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
    command = [
        "ffmpeg",
        "-i",
        input_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        temp_wav,
    ]
    try:
        subprocess.run(command, check=True)
        return filename, temp_wav
    except Exception as e:
        print(f"FFmpeg error for {filename}: {e}")
        return filename, None


def run_stage1(input_dir, output_dir, force_cpu=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    media_files = get_sorted_media_files(input_dir)
    if not media_files:
        print("No audio or video files found in input directory.")
        return

    files_to_process = []
    for filename in media_files:
        base_name = os.path.splitext(filename)[0]
        output_txt = os.path.join(output_dir, base_name + ".txt")
        if os.path.exists(output_txt):
            print(f"Skipping (Exists): {filename}")
            continue
        files_to_process.append(filename)

    if not files_to_process:
        print("Stage 1 complete (all files already processed).")
        return

    print(
        f"\n[Parallel FFmpeg] Extracting audio for {len(files_to_process)} media files..."
    )
    converted_audio_map = {}
    max_workers = min(os.cpu_count() or 4, 8)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(prepare_media_file, input_dir, fname)
            for fname in files_to_process
        ]
        for future in concurrent.futures.as_completed(futures):
            fname, temp_wav = future.result()
            if temp_wav:
                converted_audio_map[fname] = temp_wav

    if force_cpu:
        print(
            f"\nForce CPU Mode Enabled. Loading Whisper Model ({MODEL_SIZE}) on CPU (int8)..."
        )
        model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    else:
        try:
            print(f"\nLoading Whisper Model ({MODEL_SIZE}) on GPU (float16)...")
            model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
            print("Successfully loaded on GPU (float16).")
        except Exception as e1:
            print(f"float16 failed: {e1}")
            try:
                print("Trying GPU (float32)...")
                model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float32")
                print("Successfully loaded on GPU (float32).")
            except Exception as e2:
                print(f"float32 failed: {e2}")
                print("Falling back to CPU (int8)...")
                model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    for filename in files_to_process:
        clean_audio_path = converted_audio_map.get(filename)
        if not clean_audio_path:
            print(f"Skipping {filename} due to extraction failure.")
            continue

        print(f"Transcribing: {filename}")
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, base_name + ".txt")

        try:
            # VAD filter prevents hallucination on silences.
            # condition_on_previous_text prevents infinite repeating loops.
            segments, info = model.transcribe(
                clean_audio_path,
                beam_size=5,
                task="transcribe",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                condition_on_previous_text=False,
            )

            prev_text = ""
            with open(output_path, "w", encoding="utf-8") as f:
                for segment in segments:
                    text = segment.text.strip()
                    # Only write if it's not empty, and NOT an exact duplicate of the last line
                    if text and text.lower() != prev_text.lower():
                        ts_str = format_timestamp(segment.start)
                        f.write(f"{ts_str} {text}\n")
                        prev_text = text

        except Exception as e:
            print(f"Failed to process {filename}: {e}")
        finally:
            if clean_audio_path and os.path.exists(clean_audio_path):
                os.remove(clean_audio_path)

    # EXPLICITLY DELETE THE MODEL TO PREVENT STAGE 2 SEGFAULT CORE DUMPS
    print("\n[System] Unloading Whisper model from memory...")
    del model
    gc.collect()

    print("Stage 1 complete.")
