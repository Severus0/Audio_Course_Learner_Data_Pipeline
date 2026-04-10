***

# Audio Course Learner — Data Pipeline 🛠️🎧

**Audio Course Learner** is an Android application that helps users learn languages via interactive audio or video courses. It pauses at appropriate moments to capture the user’s response using **speech-to-text**, aligning user input with the original lesson media through **timestamped lesson files**.

📱 **Main App Repository:**[Audio_Course_Learner](https://github.com/Severus0/Audio_Course_Learner)

This **data pipeline** is the companion tool designed to convert raw course media into **timestamped text annotations** and package everything into a **ready-to-import `.zip`** for the Android app. It features a Graphical User Interface (GUI) and a 1-click setup.

---

## 🚨 Current Project Status: PAUSED (Seeking Contributors)

**Development of both this pipeline and the main Android app is currently paused.**

While the pipeline successfully runs and generates the required `.zip` files, **the transcription and annotation accuracy is currently hovering between 40–80%**. Because the mobile app relies on precise timestamps and exact target phrases to grade the user's speech, this current accuracy rate requires far too much manual revision in the app's built-in editor to be practical for daily use.

### The Goal: 95%+ Accuracy
Creating interactive courses requires a near-flawless data extraction process. To justify reactivating active development on this project, the pipeline needs to hit **at least 95% accuracy**—where the vast majority of expected phrases and timestamps are correctly assigned, leaving only a few odd corrections for the user.

### Can you fix this? 🤝
If you are a developer interested in **Audio Processing**, **AI Data Pipelines**, **LLM Prompt Engineering**, or advanced **Whisper** implementations, your help is incredibly welcome! 

Whether it's writing a better prompt for the LLM filtering stage, using a different transcription engine, or implementing better audio-silence detection—feel free to fork this project, experiment, and reach out. If the 95% accuracy threshold can be cracked, I will gladly resume development on the ecosystem.

---

## ✨ Key Features (Currently Implemented)

* **1-Click Launchers:** Run `start.bat` (Windows) or `start.sh` (Mac/Linux) to automatically set up a Python environment and install dependencies.
* **Graphical Interface:** No coding required. Select input and output folders, optionally provide an LLM model, and click **Run**.
* **Optional LLM Filtering:** Use a `.gguf` model (e.g., LLaMA) to automatically remove lines that are not in the target language. This stage is skipped if no model is provided.
* **Concurrent Audio/Video Processing:** Converts multiple files in parallel and extracts audio from video automatically. Supported formats include `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`.
* **Force CPU Execution:** Avoid GPU memory crashes by running Faster-Whisper and the LLM entirely on CPU.
* **Automatic Timestamped Transcription:** Uses **Faster-Whisper** to capture multilingual text.

---

## ⚙️ Pipeline Overview

The pipeline has three main stages:

### Stage 1 — Transcription (`stage_1_transcribe.py`)

* Extracts audio from audio/video files in parallel.
* Transcribes with Faster-Whisper.
* Detects and timestamps text segments.

### Stage 2 — Optional LLM Filtering (`stage_2_correct.py`)

* Skipped if no `.gguf` model is provided.
* Uses an LLM to remove lines that are not in the target language or are obvious garbage.

### Stage 3 — Packaging (`stage_3_package.py`)

* Aligns text files with media.
* Packages everything, including a `config.json` with the target language, into a `.zip` ready for the Android app.

**Processing Flow Diagram:**

```
Audio / Video
    │
    ▼
Stage 1: Whisper Transcription
    │
    ▼
Detection & Timestamped Text
    │
    ▼
Stage 2: Optional LLM Filtering
    │
    ▼
Stage 3: Packaging → ZIP for Android App
```

---

## 🛠️ Prerequisites

1. **Python 3.10+**
2. **FFmpeg** installed for audio/video extraction:

   * **Mac:** `brew install ffmpeg`
   * **Linux:** `sudo apt install ffmpeg`
   * **Windows:** Download `ffmpeg.exe` and either add it to PATH or place it next to `main.py`.
3. **NVIDIA GPU (Optional):** Faster-Whisper runs on GPU if available. CPU fallback works automatically but is slower.

---


## 🚀 Quick Start (Recommended)

You no longer need to use the command line to set up environments or install dependencies!

1. **Download/Clone this repository.**
2. **Place your course audio files** (MP3, WAV, M4A) inside the `input_audio/` folder.
3. **Run the Launcher:**
   * **Windows:** Double-click `start.bat`
   * **Mac/Linux:** Open terminal and run `bash start.sh`
4. The launcher will safely install all requirements in an isolated folder and open the **Graphical User Interface**.

5. In the GUI:

   * Select the **Target Language** which is the language to be learned, so if instructions are given in English but it's a Spanish learning course, type in 'Spanish'.
   * Select the **input folder** (audio/video) and **output folder**.
   * Set **output zip filename**.
   * (Optional) Select a `.gguf` LLM model for Stage 2 filtering.
   * Enable **Force CPU Execution** if you can't run it on GPU.
   * Click **Run Pipeline**.

---

## 📁 Directory Structure

```text
Project_Root/
├── input_audio/        # Place media files here
├── models/             # Optional .gguf LLM files
├── output/             # Final zip + temporary files
├── pipeline/           # Pipeline source code
├── main.py             # GUI & CLI orchestrator
├── start.bat           # Windows 1-click launcher
├── start.sh            # Mac/Linux 1-click launcher
└── requirements.txt
```

---

## 🐛 Troubleshooting

* **"FFmpeg not found" on Windows:**
  Drop `ffmpeg.exe` in the main folder, or ensure it is correctly added to your Environment Variables.
* **Out of Memory (VRAM) Crashes:** 
    If your GPU crashes due to low VRAM, enable "Force CPU Execution" in the GUI.
* **GPU not being used?**
  Ensure your NVIDIA drivers and CUDA toolkit are installed. You may need to reinstall `llama-cpp-python` with CUDA flags. See the "Help" button inside the GUI for exact commands.

### App License

Licensed under the **GNU General Public License v3.0 or later (GPLv3+)**.  
See the [COPYING](./COPYING) file for details.

**Author:** Seweryn Polec  
**Contact:** sewerynpolec@gmail.com  

---

## Legal Notice

This software is provided **“as is”**, without any express or implied warranty. In no event shall the author be held liable for any damages arising from the use of this software.

© 2026 Seweryn Polec. All rights reserved.
