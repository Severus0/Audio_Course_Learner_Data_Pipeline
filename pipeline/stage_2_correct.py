import os
import re
import zipfile
import json
from tqdm import tqdm
try:
    from llama_cpp import Llama
    import fasttext
except ImportError:
    print("Error: Missing dependencies. Run: pip install llama-cpp-python fasttext-wheel")
    raise

# Map standard language names to ISO codes for FastText
LANGUAGE_CODES = {
    "german": "de", "english": "en", "spanish": "es", "french": "fr",
    "italian": "it", "portuguese": "pt", "dutch": "nl", "russian": "ru",
    "chinese": "zh", "japanese": "ja", "korean": "ko"
}

def should_skip_lang_detection(text):
    # Skip short words or numbers
    letters_only = re.sub(r'[^A-Za-zÀ-ÿ]', '', text)
    return len(letters_only) < 4

def is_obvious_garbage(sentence):
    s = sentence.strip()
    return not s or len(s) < 2

def is_valid_sentence_llm(llm, sentence, language_hint="German"):
    prompt = f"""[INST]
You are a professional linguist. I will give you a sentence.
Determine if this sentence is valid, grammatical, and natural in {language_hint}.
Do NOT change it. Only answer YES if it is correct, NO if it is ungrammatical, nonsensical. Ignore foreign language concerns.
Sentence: "{sentence}"
Answer with only YES or NO.
[/INST]"""
    try:
        output = llm(prompt, max_tokens=5, stop=["\n", "."], echo=False)
        text = output["choices"][0]["text"].strip().upper()
        return "NO" not in text # Default to True if ambiguous
    except Exception as e:
        print(f"LLM Error: {e}")
        return True

def run_stage2(language_hint, transcribed_dir, input_audio_dir, zip_name, llm_model_path, fasttext_model_path):
    target_lang_code = LANGUAGE_CODES.get(language_hint.lower())
    
    # Ensure output dir for zip exists
    zip_dir = os.path.dirname(zip_name)
    if zip_dir and not os.path.exists(zip_dir):
        os.makedirs(zip_dir)

    print(f"Loading FastText model from {fasttext_model_path}...")
    lang_model = fasttext.load_model(fasttext_model_path)

    print(f"Loading LLM from {llm_model_path}...")
    llm = Llama(
        model_path=llm_model_path,
        n_ctx=2048,
        n_gpu_layers=-1, # Max GPU offload
        verbose=False
    )

    txt_files = [f for f in os.listdir(transcribed_dir) if f.endswith(".txt")]
    
    # Open Zip
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zipf:
        
        for txt_file in txt_files:
            print(f"Validating: {txt_file}")
            txt_path = os.path.join(transcribed_dir, txt_file)

            with open(txt_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            cleaned_lines = []
            seen = set()

            for line in tqdm(lines, desc="Processing Lines", leave=False):
                # Parse Timestamp
                if " " in line:
                    timestamp, text = line.split(" ", 1)
                else:
                    timestamp, text = "00:00", line

                if is_obvious_garbage(text): continue
                if text.lower() in seen: continue
                seen.add(text.lower())

                # 1. FastText Language Check
                if target_lang_code and not should_skip_lang_detection(text):
                    labels, probs = lang_model.predict(text, k=3)
                    # Normalize labels from __label__de to de
                    langs = {l.replace("__label__", ""): p for l, p in zip(labels, probs)}
                    
                    target_prob = langs.get(target_lang_code, 0)
                    # If the detected language is definitely NOT target
                    # (e.g. English 0.9, German 0.1) -> Skip
                    if target_prob < 0.20:
                        continue

                # 2. LLM Semantics Check
                if not is_valid_sentence_llm(llm, text, language_hint):
                    continue

                cleaned_lines.append(f"{timestamp} {text}")

            # Write cleaned content to Zip
            zipf.writestr(txt_file, "\n".join(cleaned_lines))

            # Find and add Audio
            base_name = os.path.splitext(txt_file)[0]
            for ext in [".mp3", ".wav", ".m4a"]:
                audio_candidate = os.path.join(input_audio_dir, base_name + ext)
                if os.path.exists(audio_candidate):
                    zipf.write(audio_candidate, arcname=base_name + ext)
                    break

        # Add config file for the App
        zipf.writestr("config.json", json.dumps({"language": language_hint}, indent=4))

    print(f"Stage 2 Complete. Zip created at {zip_name}")
