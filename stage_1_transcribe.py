import sys
import os
import subprocess
import re
from faster_whisper import WhisperModel


PAUSE_THRESHOLD = 1.3
MODEL_SIZE = "large-v3"
COMPUTE_TYPE = "float32"


def convert_audio_if_needed(input_path):
    filename = os.path.splitext(os.path.basename(input_path))[0]
    temp_wav = f"temp_{filename}.wav"

    if os.path.exists(temp_wav):
        os.remove(temp_wav)

    command = [
        "ffmpeg", "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-y",
        "-hide_banner", "-loglevel", "error",
        temp_wav
    ]

    subprocess.run(command, check=True)
    return temp_wav


def format_timestamp(seconds):
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{int(s):02d}"


def extract_lesson_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else None


def get_sorted_audio_files(input_dir):
    files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".mp3", ".wav", ".m4a", ".flac"))
    ]

    def sort_key(filename):
        number = extract_lesson_number(filename)
        if number is not None:
            return (0, number)
        return (1, filename.lower())

    return sorted(files, key=sort_key)


def process_file(model, input_dir, output_dir, filename):
    input_path = os.path.join(input_dir, filename)
    output_filename = os.path.splitext(filename)[0] + ".txt"
    output_path = os.path.join(output_dir, output_filename)

    if os.path.exists(output_path):
        print(f"Skipping (already transcribed): {filename}")
        return

    print(f"Processing: {filename}")

    clean_audio_path = convert_audio_if_needed(input_path)

    segments, info = model.transcribe(
        clean_audio_path,
        beam_size=5,
        word_timestamps=True,
        task="transcribe"
    )

    all_words = []
    for segment in segments:
        for word in segment.words:
            all_words.append(word)

    annotations = []

    for i in range(len(all_words) - 1):
        current_word = all_words[i]
        next_word = all_words[i + 1]
        gap = next_word.start - current_word.end

        if gap > PAUSE_THRESHOLD:
            phrase_parts = []

            for j in range(i + 1, len(all_words)):
                w = all_words[j]
                phrase_parts.append(w.word)

            answer_phrase = "".join(phrase_parts).strip()

            if len(answer_phrase) > 1:
                timestamp_str = format_timestamp(next_word.start)
                annotations.append(f"{timestamp_str} {answer_phrase}")

    with open(output_path, "w", encoding="utf-8") as f:
        for line in annotations:
            f.write(line + "\n")

    os.remove(clean_audio_path)


def run_stage1(input_dir, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading Whisper {MODEL_SIZE}...")
    model = WhisperModel(MODEL_SIZE, device="cuda", compute_type=COMPUTE_TYPE)

    audio_files = get_sorted_audio_files(input_dir)

    for filename in audio_files:
        process_file(model, input_dir, output_dir, filename)

    print("Stage 1 complete.")
