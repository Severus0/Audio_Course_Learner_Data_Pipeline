import argparse
import gc
import json
import os
import shutil
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk

sys.path.append(os.path.join(os.path.dirname(__file__), "pipeline"))
import stage_1_transcribe
import stage_2_correct
import stage_3_package

STATE_FILE = ".pipeline_state.json"
DEFAULT_AUDIO_DIR = "input_audio"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_ZIP_NAME = "course_processed.zip"
MODELS_DIR = "models"


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4, ensure_ascii=False)


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_default_llm():
    """Finds a default model ONLY for the first time the app is launched."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    gguf_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".gguf")]
    if gguf_files:
        return os.path.join(MODELS_DIR, gguf_files[0])
    return ""


def resolve_llm_path(explicit_path):
    """Strictly respects the user's input. If empty, skips Stage 2."""
    if not explicit_path or not str(explicit_path).strip():
        print(
            "\n[NOTE] LLM path is empty. Stage 2 (Language Filtering) will be skipped."
        )
        return None

    if os.path.isfile(explicit_path):
        print(f"Note: Using LLM model '{explicit_path}'")
        return explicit_path
    else:
        print(
            f"\n[WARNING] LLM model not found at '{explicit_path}'. Stage 2 will be skipped."
        )
        return None


def force_cleanup():
    print("[System] Clearing memory...")
    gc.collect()
    time.sleep(2)


def run_pipeline(state, llm_path):
    print("\n=== Stage 1: Full Transcription ===\n")
    try:
        stage_1_transcribe.run_stage1(
            input_dir=state["audio_dir"],
            output_dir=state["transcribed_dir"],
            force_cpu=state.get("force_cpu", False),
        )
    except Exception as e:
        print(f"Stage 1 Error: {e}")
        return

    force_cleanup()

    final_text_dir = state["transcribed_dir"]

    if llm_path:
        print("\n=== Stage 2: LLM Language Filtering ===\n")
        try:
            stage_2_correct.run_stage2(
                language_hint=state["language"],
                transcribed_dir=state["transcribed_dir"],
                corrected_dir=state["corrected_dir"],
                llm_model_path=llm_path,
                force_cpu=state.get("force_cpu", False),
            )
            final_text_dir = state["corrected_dir"]
        except Exception as e:
            print(f"Stage 2 Error: {e}")
            return
        force_cleanup()
    else:
        print("\n=== Stage 2 Skipped (No LLM Provided) ===\n")

    print("\n=== Stage 3: Packaging Course ===\n")
    try:
        stage_3_package.run_stage3(
            text_dir=final_text_dir,
            audio_dir=state["audio_dir"],
            zip_name=state["zip_name"],
            language=state["language"],
        )
        print(f"\nSUCCESS! File created: {state['zip_name']}")
    except Exception as e:
        print(f"Stage 3 Error: {e}")

    if os.path.exists(state["corrected_dir"]):
        shutil.rmtree(state["corrected_dir"], ignore_errors=True)
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)


class RedirectText(object):
    def __init__(self, text_ctrl):
        self.output = text_ctrl

    def write(self, string):
        self.output.insert(tk.END, string)
        self.output.see(tk.END)

    def flush(self):
        pass


class AppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Language Course Annotator (Audio & Video)")
        self.root.geometry("750x700")
        state = load_state()
        self.lang_var = tk.StringVar(value=state.get("language", "German"))
        self.input_var = tk.StringVar(value=state.get("audio_dir", DEFAULT_AUDIO_DIR))
        self.output_var = tk.StringVar(
            value=state.get("output_dir", DEFAULT_OUTPUT_DIR)
        )
        self.zipname_var = tk.StringVar(value=state.get("zip_name", DEFAULT_ZIP_NAME))
        self.force_cpu_var = tk.BooleanVar(value=state.get("force_cpu", False))

        # Uses default LLM ONLY if state has no memory of this field at all
        self.llm_var = tk.StringVar(value=state.get("llm_path", get_default_llm()))

        self._build_ui()
        sys.stdout = RedirectText(self.log_text)
        sys.stderr = RedirectText(self.log_text)

    def _build_ui(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        # Row 0
        ttk.Label(frame, text="Target Language:").grid(
            row=0, column=0, sticky="w", pady=5
        )
        ttk.Entry(frame, textvariable=self.lang_var).grid(
            row=0, column=1, sticky="ew", pady=5
        )

        # Row 1
        ttk.Label(frame, text="Input Media (Audio/Video):").grid(
            row=1, column=0, sticky="w", pady=5
        )
        ttk.Entry(frame, textvariable=self.input_var).grid(
            row=1, column=1, sticky="ew", pady=5
        )
        ttk.Button(
            frame, text="Browse", command=lambda: self.browse_dir(self.input_var)
        ).grid(row=1, column=2, padx=5)

        # Row 2
        ttk.Label(frame, text="Output Folder:").grid(
            row=2, column=0, sticky="w", pady=5
        )
        ttk.Entry(frame, textvariable=self.output_var).grid(
            row=2, column=1, sticky="ew", pady=5
        )
        ttk.Button(
            frame, text="Browse", command=lambda: self.browse_dir(self.output_var)
        ).grid(row=2, column=2, padx=5)

        # Row 3
        ttk.Label(frame, text="Output Zip Filename:").grid(
            row=3, column=0, sticky="w", pady=5
        )
        ttk.Entry(frame, textvariable=self.zipname_var).grid(
            row=3, column=1, sticky="ew", pady=5
        )

        # Row 4
        ttk.Label(frame, text="LLM Model (.gguf) [Optional]:").grid(
            row=4, column=0, sticky="w", pady=5
        )
        ttk.Entry(frame, textvariable=self.llm_var).grid(
            row=4, column=1, sticky="ew", pady=5
        )
        ttk.Button(
            frame, text="Browse File", command=lambda: self.browse_file(self.llm_var)
        ).grid(row=4, column=2, padx=5)

        # Row 5 — options frame
        options_frame = ttk.Frame(frame)
        options_frame.grid(row=5, column=0, columnspan=3, sticky="w", pady=10)
        ttk.Checkbutton(
            options_frame,
            text="Force CPU Execution (Slower, fixes GPU memory crashes)",
            variable=self.force_cpu_var,
        ).pack(side=tk.TOP, anchor="w", pady=2)

        # Row 6 — buttons frame
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=6, column=0, columnspan=3, pady=15)
        ttk.Button(
            btn_frame, text="Run Pipeline", command=self.start_pipeline, width=20
        ).pack(side=tk.LEFT, padx=10)
        ttk.Button(
            btn_frame, text="Help / GPU Instructions", command=self.show_help
        ).pack(side=tk.LEFT, padx=10)

        # Row 7 & 8 — activity log
        ttk.Label(frame, text="Activity Log:").grid(row=7, column=0, sticky="w")
        self.log_text = scrolledtext.ScrolledText(
            frame, height=20, bg="black", fg="lightgray"
        )
        self.log_text.grid(row=8, column=0, columnspan=3, sticky="nsew")

        frame.rowconfigure(8, weight=1)
        frame.columnconfigure(1, weight=1)

    def browse_dir(self, var):
        folder = filedialog.askdirectory()
        if folder:
            var.set(folder)

    def browse_file(self, var):
        filepath = filedialog.askopenfilename(
            filetypes=[("GGUF Models", "*.gguf"), ("All Files", "*.*")]
        )
        if filepath:
            var.set(filepath)

    def show_help(self):
        help_win = tk.Toplevel(self.root)
        help_win.title("Help, Requirements & GPU Instructions")
        help_win.geometry("700x600")
        txt = scrolledtext.ScrolledText(help_win, wrap=tk.WORD, font=("Consolas", 10))
        txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        help_text = """HOW IT WORKS (SIMPLIFIED PIPELINE):
1. Put your AUDIO or VIDEO files in the Input folder.
2. Whisper transcribes the ENTIRE file line-by-line.
3. (OPTIONAL): If you provide an LLM, it scans each sentence. If it spots English instruction phrases, it safely deletes them to save you editing time (isn't 100% effective, so some English sentence artefacts may remain).
4. The final Zip contains all the timestamps. Open it in the Android app to easily delete the ones you don't want!

CPU vs GPU EXECUTION:
Check "Force CPU Execution" if your GPU is crashing due to low VRAM.

HOW TO ENABLE GPU ACCELERATION:
Note: Installing CUDA and building with GPU support requires matching CUDA/cuDNN versions.
[Windows - PowerShell]
pip uninstall -y llama-cpp-python
setx CMAKE_ARGS "-DGGML_CUDA=on"
pip install llama-cpp-python

[Linux / macOS - Bash]
pip uninstall -y llama-cpp-python
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
"""
        txt.insert(tk.END, help_text)
        txt.config(state=tk.DISABLED)

    def start_pipeline(self):
        out_dir = self.output_var.get()
        state = {
            "language": self.lang_var.get(),
            "audio_dir": self.input_var.get(),
            "output_dir": out_dir,
            "llm_path": self.llm_var.get(),
            "transcribed_dir": os.path.join(out_dir, "transcribed"),
            "corrected_dir": os.path.join(out_dir, "corrected_temp"),
            "zip_name": os.path.join(
                out_dir, self.zipname_var.get() or DEFAULT_ZIP_NAME
            ),
            "force_cpu": self.force_cpu_var.get(),
        }
        if not os.path.exists(state["audio_dir"]):
            os.makedirs(state["audio_dir"])
            print(f"Created '{state['audio_dir']}'. Please put your media files there.")
            return
        save_state(state)
        threading.Thread(target=self._run, args=(state,), daemon=True).start()

    def _run(self, state):
        llm_path = resolve_llm_path(state.get("llm_path"))
        run_pipeline(state, llm_path)


def main_cli():
    state = load_state()
    language = input("Language being learned? [German]: ").strip() or "German"
    audio_dir = (
        input(f"Source media directory[{DEFAULT_AUDIO_DIR}]: ").strip()
        or DEFAULT_AUDIO_DIR
    )
    out_dir = (
        input(f"Output directory[{DEFAULT_OUTPUT_DIR}]: ").strip() or DEFAULT_OUTPUT_DIR
    )

    zip_name_input = input(f"Output zip filename [{DEFAULT_ZIP_NAME}]: ").strip()
    zip_name = zip_name_input if zip_name_input else DEFAULT_ZIP_NAME

    default_llm = get_default_llm()
    prompt_str = (
        f"LLM .gguf Path [{default_llm}] (Leave blank to skip LLM validation): "
    )
    llm_path_in = input(prompt_str).strip()

    # If they just press enter, they want the default. If they type "skip" or space, skip.
    llm_path = llm_path_in if llm_path_in else default_llm

    force_cpu_in = input("Force CPU execution? (y/N): ").strip().lower() == "y"

    state.update(
        {
            "language": language,
            "audio_dir": audio_dir,
            "llm_path": llm_path,
            "transcribed_dir": os.path.join(out_dir, "transcribed"),
            "corrected_dir": os.path.join(out_dir, "corrected_temp"),
            "zip_name": os.path.join(out_dir, zip_name),
            "force_cpu": force_cpu_in,
        }
    )
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
        print(f"Created '{audio_dir}'. Put media files there and restart.")
        sys.exit(0)
    save_state(state)
    resolved_path = resolve_llm_path(state.get("llm_path"))
    run_pipeline(state, resolved_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cli", action="store_true", help="Run in Command Line mode instead of GUI"
    )
    args = parser.parse_args()
    if args.cli:
        main_cli()
    else:
        root = tk.Tk()
        app = AppGUI(root)
        root.mainloop()
