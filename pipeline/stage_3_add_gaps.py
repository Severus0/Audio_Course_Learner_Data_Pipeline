import json
import os
import subprocess
import zipfile

from faster_whisper import WhisperModel

# --- CONFIG ---
GAP_THRESHOLD = 0.7
COLLISION_WINDOW = 1.5
MODEL_SIZE = "large-v3"
COMPUTE_TYPE = "float32"


def time_to_seconds(time_str):
    try:
        parts = time_str.strip().split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return 0.0
    except:
        return 0.0


def format_timestamp(seconds):
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{int(s):02d}"


def convert_audio_if_needed(input_path):
    filename = os.path.splitext(os.path.basename(input_path))[0]
    temp_wav = os.path.join(os.path.dirname(input_path), f".temp_s3_{filename}.wav")
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
        return temp_wav
    except:
        return None


def run_stage3(corrected_dir, audio_dir, zip_name, language):
    # Ensure zip directory exists
    zip_folder = os.path.dirname(zip_name)
    if zip_folder and not os.path.exists(zip_folder):
        os.makedirs(zip_folder)

    print(f"Loading Whisper Model ({MODEL_SIZE})...")
    try:
        model = WhisperModel(MODEL_SIZE, device="cuda", compute_type=COMPUTE_TYPE)
    except Exception:
        print("GPU Load failed, falling back to CPU...")
        model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    # Get all corrected text files
    txt_files = [f for f in os.listdir(corrected_dir) if f.endswith(".txt")]

    print(f"Creating Zip: {zip_name}")
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zout:

        # Add Config
        zout.writestr("config.json", json.dumps({"language": language}, indent=4))

        for txt_file in txt_files:
            print(f"Processing: {txt_file}")

            # 1. Read Corrected Text
            txt_path = os.path.join(corrected_dir, txt_file)
            existing_lines = []

            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(" ", 1)
                    secs = time_to_seconds(parts[0])
                    existing_lines.append((secs, line))

            # 2. Find and Process Audio
            base_name = os.path.splitext(txt_file)[0]
            audio_path = None
            found_audio_name = None

            for ext in [".mp3", ".wav", ".m4a", ".flac", ".ogg"]:
                cand = os.path.join(audio_dir, base_name + ext)
                if os.path.exists(cand):
                    audio_path = cand
                    found_audio_name = base_name + ext
                    break

            final_lines = existing_lines.copy()

            if audio_path:
                # Add Audio to Zip
                zout.write(audio_path, arcname=found_audio_name)

                # Run Whisper for Gaps
                clean_audio = convert_audio_if_needed(audio_path)
                if clean_audio:
                    try:
                        segments, _ = model.transcribe(
                            clean_audio, word_timestamps=True
                        )
                        all_words = [w for s in segments for w in s.words]

                        candidate_gaps = []
                        for i in range(len(all_words) - 1):
                            curr_w = all_words[i]
                            next_w = all_words[i + 1]
                            gap = next_w.start - curr_w.end

                            if gap > GAP_THRESHOLD:
                                candidate_gaps.append(next_w.start)

                        # Merge Gaps
                        for gap_time in candidate_gaps:
                            is_covered = False
                            for ex_time, _ in existing_lines:
                                if abs(gap_time - ex_time) < COLLISION_WINDOW:
                                    is_covered = True
                                    break

                            if not is_covered:
                                ts_str = format_timestamp(gap_time)
                                final_lines.append((gap_time, f"{ts_str}"))

                    except Exception as e:
                        print(f" - Error processing audio for gaps: {e}")
                    finally:
                        if os.path.exists(clean_audio):
                            os.remove(clean_audio)
            else:
                print(f" - Warning: No audio found for {txt_file}")

            # 3. Sort and Write to Zip
            final_lines.sort(key=lambda x: x[0])
            new_content = "\n".join([x[1] for x in final_lines])
            zout.writestr(txt_file, new_content)

    print("Stage 3 Complete: Gaps added and Zip created.")
