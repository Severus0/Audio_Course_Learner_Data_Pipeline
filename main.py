import argparse
import gc
import json
import os
import shutil
import sys
import time

# --- AUTO-FIX CUDA PATHS (For Linux/NVIDIA users) ---
try:
    import nvidia.cublas.lib
    import nvidia.cudnn.lib

    cublas_path = os.path.dirname(nvidia.cublas.lib.__file__)
    cudnn_path = os.path.dirname(nvidia.cudnn.lib.__file__)
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{current_ld}:{cublas_path}:{cudnn_path}"
except ImportError:
    pass

# Adjust path to find pipeline modules
sys.path.append(os.path.join(os.path.dirname(__file__), "pipeline"))

import stage_1_transcribe
import stage_2_correct
import stage_3_add_gaps

# --- CONSTANTS ---
STATE_FILE = ".pipeline_state.json"
DEFAULT_AUDIO_DIR = "input_audio"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_ZIP_NAME = "course_processed.zip"
MODELS_DIR = "models"
DEFAULT_FASTTEXT = "lid.176.bin"


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4, ensure_ascii=False)


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def check_models_exist():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    ft_path = os.path.join(MODELS_DIR, DEFAULT_FASTTEXT)

    missing = []
    if not os.path.exists(ft_path):
        missing.append(f"FastText model missing: {ft_path}")

    gguf_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".gguf")]
    if not gguf_files:
        missing.append(f"Llama GGUF model missing in '{MODELS_DIR}/'")
        final_llm_path = None
    else:
        final_llm_path = os.path.join(MODELS_DIR, gguf_files[0])
        print(f"Note: Using found model '{gguf_files[0]}'")

    if missing:
        print("\n!!! MISSING MODELS !!!")
        for m in missing:
            print(f" - {m}")
        return None, None

    return ft_path, final_llm_path


def ask_with_default(prompt_text, default_value):
    user_input = input(f"{prompt_text} [{default_value}]: ").strip()
    return user_input if user_input else default_value


def get_configuration(resume=False):
    if resume:
        state = load_state()
        if state:
            return state

    print("\n=== Language Course Processing Pipeline ===\n")
    language = input(
        "What language is being learned? (e.g., German, Spanish): "
    ).strip()
    if not language:
        sys.exit(1)

    if not os.path.exists(DEFAULT_AUDIO_DIR):
        os.makedirs(DEFAULT_AUDIO_DIR)
        print(f"Created '{DEFAULT_AUDIO_DIR}'. Put audio files there and restart.")
        sys.exit(0)

    audio_dir = ask_with_default("Source directory", DEFAULT_AUDIO_DIR)
    output_dir = ask_with_default("Output directory", DEFAULT_OUTPUT_DIR)
    zip_name = ask_with_default("Zip archive name", DEFAULT_ZIP_NAME)

    state = {
        "language": language,
        "audio_dir": audio_dir,
        "transcribed_dir": os.path.join(output_dir, "transcribed"),
        "corrected_dir": os.path.join(output_dir, "corrected_temp"),
        "zip_name": os.path.join(output_dir, zip_name),
    }

    save_state(state)
    return state


def force_cleanup():
    """Forces Python to release memory and sleep briefly to allow VRAM clear."""
    print("   [System] Clearing memory...")
    gc.collect()
    time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    ft_path, llm_path = check_models_exist()
    if not ft_path:
        sys.exit(1)

    if args.reset:
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
        sys.exit(0)

    state = get_configuration(resume=args.resume)

    print("\n--- Configuration ---")
    print(f"Target: {state['language']}")
    print(f"Input:  {state['audio_dir']}")
    print(f"LLM:    {os.path.basename(llm_path)}")
    print("---------------------\n")

    if not args.resume:
        if input("Start? (y/n): ").strip().lower() != "y":
            sys.exit(0)

    # --- STAGE 1: Transcription ---
    print("\n=== Stage 1: Transcription ===\n")
    try:
        stage_1_transcribe.run_stage1(
            input_dir=state["audio_dir"], output_dir=state["transcribed_dir"]
        )
    except Exception as e:
        print(f"Stage 1 Error: {e}")
        sys.exit(1)

    force_cleanup()

    # --- STAGE 2: Correction (To Temp Folder) ---
    print("\n=== Stage 2: Correction & Validation ===\n")
    try:
        stage_2_correct.run_stage2(
            language_hint=state["language"],
            transcribed_dir=state["transcribed_dir"],
            corrected_dir=state["corrected_dir"],  # Save to temp folder, not Zip
            llm_model_path=llm_path,
            fasttext_model_path=ft_path,
        )
    except Exception as e:
        print(f"Stage 2 Error: {e}")
        sys.exit(1)

    force_cleanup()

    # --- STAGE 3: Add Gaps & Create Zip ---
    print("\n=== Stage 3: Adding Pauses & Packaging ===\n")
    try:
        stage_3_add_gaps.run_stage3(
            corrected_dir=state["corrected_dir"],
            audio_dir=state["audio_dir"],
            zip_name=state["zip_name"],
            language=state["language"],
        )
    except Exception as e:
        print(f"Stage 3 Error: {e}")
        sys.exit(1)

    # Clean up temp corrected folder
    if os.path.exists(state["corrected_dir"]):
        shutil.rmtree(state["corrected_dir"])

    print(f"\nSUCCESS! File created: {state['zip_name']}")
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)


if __name__ == "__main__":
    main()
