# Audio Course Learner Data Pipeline

**Audio_Course_Learner** is an Android application designed to help users learn languages through interactive audio courses. Many existing courses ask a question and then pause, expecting the user to respond verbally. However, there is typically no way to verify whether the user's answer or pronunciation is correct.

The **Audio Course Learner** app solves this by stopping playback at the appropriate moments and capturing the user's response via speech-to-text in the target language. It relies on **timestamped lesson files** that align with the course audio.

This **data pipeline** automatically processes audio courses, generating timestamped transcription files and a packaged `.zip` archive ready for import into the app. This allows enthusiasts to prepare courses on their desktop and use them seamlessly in the Android application.

---

## Pipeline Overview

The pipeline is organized into a modular structure with a central orchestrator:

* **main.py:**  
  The entry point. It handles configuration, checks for required models, auto-configures GPU libraries (on Linux), and manages the flow between transcription and validation. It also manages memory cleanup to prevent GPU crashes between stages.

* **Stage 1: Transcription (`pipeline/stage_1_transcribe.py`):**  
  Converts audio to 16kHz WAV and uses **Faster-Whisper** (CTranslate2) to transcribe the audio.
  - **Pause Detection:** It analyzes word-level timestamps to detect silence gaps (default ~1.3s - 1.5s).
  - **Logic:** *Teacher speaks -> Silence -> Student answers*. The script captures the phrase immediately following the silence as the "Expected Answer."

* **Stage 2: Validation & Packaging (`pipeline/stage_2_correct.py`):**  
  Validates the raw transcripts using two AI models:
  1. **FastText:** Filters out sentences that do not match the target language (e.g., removing English instructions from a German course).
  2. **Llama (LLM):** Checks if the sentence is grammatically valid and natural.
  - **Packaging:** Creates a `.zip` file containing the clean text, the audio, and a `config.json` file.

---

## Directory Structure

To run the pipeline, your folder should look like this:

```text
Project_Root/
├── input_audio/         # Place your MP3/WAV course files here
├── models/              # Place AI models here (see Setup)
│   ├── lid.176.bin
│   └── llama-model.gguf
├── output/              # Generated results appear here
├── pipeline/            # Source code
├── main.py              # Run this script
└── requirements.txt
```

---

## Prerequisites

1. **Python 3.10+**
2. **FFmpeg**: Must be installed and accessible in your system `PATH`.
   - *Linux:* `sudo apt install ffmpeg`
   - *Windows:* Download binaries and add to Path.
   - *Mac:* `brew install ffmpeg`
3. **NVIDIA GPU (Recommended):** The pipeline is optimized for CUDA. It will fallback to CPU if necessary, but this is significantly slower.

---

## Setup & Usage

### 1. Install Dependencies
```bash
git clone https://github.com/Severus0/Audio_Course_Learner_Data_Pipeline
cd Audio_Course_Learner_Data_Pipeline

pip install -r requirements.txt
```

### 2. Download Models (Required)
You must download two models and place them in the `models/` folder:

1.  **FastText Language ID:**
    *   Download `lid.176.bin` from [Facebook/FastText](https://fasttext.cc/docs/en/language-identification.html).
    *   Place it at: `models/lid.176.bin`

2.  **LLM (GGUF format):**
    *   Download a generic instruction-tuned model (e.g., Llama-3 or Llama-2) in GGUF format.
    *   Recommended: [Llama-3-8B-Instruct-Q4_K_M.gguf](https://huggingface.co/bstrazzle/Meta-Llama-3-8B-Instruct-GGUF) or similar.
    *   Place it in `models/`. The script will automatically detect any `.gguf` file in that folder.

### 3. Run the Pipeline
Place your course audio files (MP3, WAV, M4A) inside the `input_audio/` folder.

```bash
python main.py
```

Follow the on-screen prompts:
1.  **Language:** Enter the target language (e.g., "German").
2.  **Start:** The pipeline will transcribe (Stage 1), clear memory, and then validate (Stage 2).

### 4. Output
Once complete, you will find a `.zip` file in the `output/` directory (e.g., `course_processed.zip`). 
Transfer this file to your phone and import it into the Audio Course Learner app.

---

## Troubleshooting

*   **"Unable to avoid copy while creating an array"**: This is caused by NumPy 2.0. Run `pip install "numpy<2.0"`.
*   **"libcublas.so.12 not found"**: The script attempts to auto-fix this. If it fails, ensure you installed the requirements and that your NVIDIA drivers are up to date.
*   **Out of Memory (VRAM)**: The pipeline attempts to clear memory between stages. If you still crash on Stage 1, edit `pipeline/stage_1_transcribe.py` and change `COMPUTE_TYPE` to `"int8"`.

### App License

Licensed under the **GNU General Public License v3.0 or later (GPLv3+)**.  
See the [COPYING](./COPYING) file for details.

**Author:** Seweryn Polec  
**Contact:** sewerynpolec@gmail.com  

---

## Legal Notice

This software is provided **“as is”**, without any express or implied warranty. In no event shall the author be held liable for any damages arising from the use of this software.

© 2026 Seweryn Polec. All rights reserved.

```
