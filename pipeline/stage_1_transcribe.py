import os
import re
import subprocess

from faster_whisper import WhisperModel

# Constants
PAUSE_THRESHOLD = 1.3
MODEL_SIZE = "large-v3"
COMPUTE_TYPE = "float32"


def convert_audio_if_needed(input_path):
    """Converts audio to 16kHz mono wav using ffmpeg."""
    filename = os.path.splitext(os.path.basename(input_path))[0]
    # Create temp file in the same directory as input to avoid cross-drive issues
    temp_wav = os.path.join(os.path.dirname(input_path), f".temp_{filename}.wav")

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
    except subprocess.CalledProcessError:
        print("Error: FFmpeg failed. Please ensure FFmpeg is installed.")
        raise
    except FileNotFoundError:
        print("Error: FFmpeg not found in system PATH.")
        raise


def format_timestamp(seconds):
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{int(s):02d}"


def get_sorted_audio_files(input_dir):
    files = [
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith((".mp3", ".wav", ".m4a", ".flac", ".ogg"))
    ]

    # Sort by number if possible (Lesson 1, Lesson 2), else alphabetical
    def sort_key(filename):
        numbers = re.findall(r"\d+", filename)
        if numbers:
            return int(numbers[0])
        return filename.lower()

    return sorted(files, key=sort_key)


def process_file(model, input_dir, output_dir, filename):
    input_path = os.path.join(input_dir, filename)
    output_filename = os.path.splitext(filename)[0] + ".txt"
    output_path = os.path.join(output_dir, output_filename)

    if os.path.exists(output_path):
        print(f"Skipping (Exists): {filename}")
        return

    print(f"Transcribing: {filename}")
    clean_audio_path = None

    try:
        clean_audio_path = convert_audio_if_needed(input_path)

        # Transcribe
        segments, info = model.transcribe(
            clean_audio_path, beam_size=5, word_timestamps=True, task="transcribe"
        )

        all_words = []
        for segment in segments:
            for word in segment.words:
                all_words.append(word)

        annotations = []
        # Pause Detection Logic
        for i in range(len(all_words) - 1):
            curr_w = all_words[i]
            next_w = all_words[i + 1]
            gap = next_w.start - curr_w.end

            if gap > PAUSE_THRESHOLD:
                # Extract the phrase after the gap
                phrase_parts = []
                for j in range(i + 1, len(all_words)):
                    w = all_words[j]

                    # Stop logic: next gap or punctuation
                    if j > i + 1:
                        prev = all_words[j - 1]
                        if (w.start - prev.end) > 1.0:
                            break
                        if prev.word.strip().endswith((".", "?", "!")):
                            break

                    phrase_parts.append(w.word)

                answer_phrase = "".join(phrase_parts).strip()

                # Clean leading punctuation
                if answer_phrase.startswith((".", ",", "?", "!")):
                    answer_phrase = answer_phrase[1:].strip()

                if len(answer_phrase) > 1:
                    ts_str = format_timestamp(next_w.start)
                    annotations.append(f"{ts_str} {answer_phrase}")

        with open(output_path, "w", encoding="utf-8") as f:
            for line in annotations:
                f.write(line + "\n")

    except Exception as e:
        print(f"Failed to process {filename}: {e}")
    finally:
        if clean_audio_path and os.path.exists(clean_audio_path):
            os.remove(clean_audio_path)


def run_stage1(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading Whisper Model ({MODEL_SIZE})...")
    # GPU check handled by caller or auto-fallback
    try:
        model = WhisperModel(MODEL_SIZE, device="cuda", compute_type=COMPUTE_TYPE)
    except Exception:
        print("GPU Load failed, falling back to CPU (int8)...")
        model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    audio_files = get_sorted_audio_files(input_dir)
    if not audio_files:
        print("No audio files found in input directory.")
        return

    for filename in audio_files:
        process_file(model, input_dir, output_dir, filename)

    print("Stage 1 complete.")
