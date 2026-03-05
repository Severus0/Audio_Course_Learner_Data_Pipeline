import os
import sys
import json
import argparse
from pathlib import Path

import stage_1_transcribe
import stage_2_correct

# --- STATE FILE ---
STATE_FILE = ".pipeline_state.json"

# --- DEFAULTS ---
DEFAULT_AUDIO_DIR = "input_audio"
DEFAULT_TRANSCRIBED_DIR = "transcribed_lessons"
DEFAULT_ZIP_NAME = "processed.zip"


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4, ensure_ascii=False)


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def ask_with_default(prompt_text, default_value):
    user_input = input(f"{prompt_text} [{default_value}]: ").strip()
    return user_input if user_input else default_value


def get_configuration(resume=False):
    if resume:
        state = load_state()
        if not state:
            print("No previous state found. Starting fresh.")
            return None
        print("Loaded previous pipeline state.")
        return state

    # New run: ask user
    print("\n=== Language Course Processing Pipeline ===\n")
    language = input("What language is being learned? (e.g., German, French): ").strip()
    if not language:
        print("Language is required.")
        sys.exit(1)

    audio_dir = ask_with_default("Source directory of audio files", DEFAULT_AUDIO_DIR)
    transcribed_dir = ask_with_default("Directory for transcribed text output", DEFAULT_TRANSCRIBED_DIR)
    zip_name = ask_with_default("Desired zip archive name", DEFAULT_ZIP_NAME)

    state = {
        "language": language,
        "audio_dir": audio_dir,
        "transcribed_dir": transcribed_dir,
        "zip_name": zip_name
    }

    save_state(state)
    return state


def main():
    parser = argparse.ArgumentParser(description="Language course processing pipeline")
    parser.add_argument("--resume", action="store_true", help="Resume previous interrupted run")
    parser.add_argument("--reset", action="store_true", help="Reset pipeline (delete state and .done files)")
    args = parser.parse_args()

    if args.reset:
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
        # Remove .done files from previous runs
        for folder in [DEFAULT_TRANSCRIBED_DIR]:
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    if f.endswith(".done"):
                        os.remove(os.path.join(folder, f))
        print("Pipeline reset complete.")
        sys.exit(0)

    # Load or ask configuration
    state = get_configuration(resume=args.resume)
    if not state:
        state = get_configuration(resume=False)

    print("\n--- Configuration ---")
    print(f"Language: {state['language']}")
    print(f"Audio source: {state['audio_dir']}")
    print(f"Transcribed output: {state['transcribed_dir']}")
    print(f"Zip archive: {state['zip_name']}")
    print("---------------------\n")

    if not args.resume:
        confirm = input("Proceed? (y/n): ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            sys.exit(0)

    # --- STAGE 1: Transcription ---
    print("\n=== Stage 1: Transcription ===\n")
    stage_1_transcribe.run_stage1(
        input_dir=state["audio_dir"],
        output_dir=state["transcribed_dir"]
    )

    # --- STAGE 2: Validation + Packaging ---
    print("\n=== Stage 2: Validation + Packaging ===\n")
    stage_2_correct.run_stage2(
        language_hint=state["language"],
        transcribed_dir=state["transcribed_dir"],
        input_audio_dir=state["audio_dir"],
        zip_name=state["zip_name"]
    )

    print("\nPipeline complete.")
    print(f"Final archive created: {state['zip_name']}")
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)


if __name__ == "__main__":
    main()
