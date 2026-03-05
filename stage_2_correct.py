import os
import re
import zipfile
import json
from tqdm import tqdm
from llama_cpp import Llama
import fasttext


LANGUAGE_CODES = {
    "afrikaans": "af",
    "albanian": "sq",
    "amharic": "am",
    "arabic": "ar",
    "armenian": "hy",
    "azerbaijani": "az",
    "basque": "eu",
    "belarusian": "be",
    "bengali": "bn",
    "bosnian": "bs",
    "bulgarian": "bg",
    "catalan": "ca",
    "cebuano": "ceb",
    "chichewa": "ny",
    "chinese": "zh",
    "corsican": "co",
    "croatian": "hr",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "esperanto": "eo",
    "estonian": "et",
    "filipino": "tl",
    "finnish": "fi",
    "french": "fr",
    "frisian": "fy",
    "galician": "gl",
    "georgian": "ka",
    "german": "de",
    "greek": "el",
    "gujarati": "gu",
    "haitian creole": "ht",
    "hausa": "ha",
    "hawaiian": "haw",
    "hebrew": "he",
    "hindi": "hi",
    "hmong": "hmn",
    "hungarian": "hu",
    "icelandic": "is",
    "igbo": "ig",
    "indonesian": "id",
    "irish": "ga",
    "italian": "it",
    "japanese": "ja",
    "javanese": "jv",
    "kannada": "kn",
    "kazakh": "kk",
    "khmer": "km",
    "korean": "ko",
    "kurdish": "ku",
    "kyrgyz": "ky",
    "lao": "lo",
    "latin": "la",
    "latvian": "lv",
    "lithuanian": "lt",
    "luxembourgish": "lb",
    "macedonian": "mk",
    "malagasy": "mg",
    "malay": "ms",
    "malayalam": "ml",
    "maltese": "mt",
    "maori": "mi",
    "marathi": "mr",
    "mongolian": "mn",
    "myanmar (burmese)": "my",
    "nepali": "ne",
    "norwegian": "no",
    "odia (oriya)": "or",
    "pashto": "ps",
    "persian": "fa",
    "polish": "pl",
    "portuguese": "pt",
    "punjabi": "pa",
    "romanian": "ro",
    "russian": "ru",
    "samoan": "sm",
    "scots gaelic": "gd",
    "serbian": "sr",
    "sesotho": "st",
    "shona": "sn",
    "sindhi": "sd",
    "sinhala": "si",
    "slovak": "sk",
    "slovenian": "sl",
    "somali": "so",
    "spanish": "es",
    "sundanese": "su",
    "swahili": "sw",
    "swedish": "sv",
    "tajik": "tg",
    "tamil": "ta",
    "telugu": "te",
    "thai": "th",
    "turkish": "tr",
    "ukrainian": "uk",
    "urdu": "ur",
    "uyghur": "ug",
    "uzbek": "uz",
    "vietnamese": "vi",
    "welsh": "cy",
    "xhosa": "xh",
    "yiddish": "yi",
    "yoruba": "yo",
    "zulu": "zu"
}


def should_skip_lang_detection(text):
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
        output = llm(prompt, max_tokens=10, temperature=0)
        text = output["choices"][0]["text"].strip().upper()
        return "YES" in text
    except Exception:
        return True


def find_matching_audio(txt_filename, input_audio_dir):
    base_name = os.path.splitext(txt_filename)[0]

    for ext in (".mp3", ".wav", ".m4a", ".flac"):
        candidate = os.path.join(input_audio_dir, base_name + ext)
        if os.path.exists(candidate):
            return candidate

    return None


def run_stage2(language_hint,
               transcribed_dir,
               input_audio_dir,
               zip_name,
               llm_model_path,
               fasttext_model_path):

    target_lang_code = LANGUAGE_CODES.get(language_hint.lower())
    if not target_lang_code:
        print("Unsupported language.")
        return

    print("Loading fastText...")
    lang_model = fasttext.load_model(fasttext_model_path)

    print("Loading LLM...")
    llm = Llama(
        model_path=llm_model_path,
        n_ctx=1024,
        n_gpu_layers=-1,
        verbose=False
    )

    txt_files = [
        f for f in os.listdir(transcribed_dir)
        if f.lower().endswith(".txt")
    ]

    if os.path.exists(zip_name):
        os.remove(zip_name)

    zipf = zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED)

    for txt_file in txt_files:

        done_marker = os.path.join(transcribed_dir, txt_file + ".done")

        if os.path.exists(done_marker):
            print(f"Skipping (already validated): {txt_file}")
            zipf.write(os.path.join(transcribed_dir, txt_file),
                       arcname=txt_file)
            continue

        print(f"Validating: {txt_file}")

        txt_path = os.path.join(transcribed_dir, txt_file)

        with open(txt_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        cleaned_lines = []
        seen = set()

        for line in tqdm(lines, leave=False):

            if " " in line:
                timestamp, text = line.split(" ", 1)
            else:
                timestamp, text = "00:00", line

            if is_obvious_garbage(text):
                continue

            if text.lower() in seen:
                continue
            seen.add(text.lower())

            # fastText filter
            if not should_skip_lang_detection(text):
                labels, probs = lang_model.predict(text, k=3)
                langs = {l.replace("__label__", ""): p
                         for l, p in zip(labels, probs)}
                target_prob = langs.get(target_lang_code, 0)
                max_prob = max(probs)
                if not (target_prob >= 0.40 and
                        (max_prob - target_prob) < 0.30):
                    continue

            if not is_valid_sentence_llm(llm, text, language_hint):
                continue

            cleaned_lines.append(f"{timestamp} {text}")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(cleaned_lines))

        zipf.write(txt_path, arcname=txt_file)

        audio_path = find_matching_audio(txt_file, input_audio_dir)
        if audio_path:
            zipf.write(audio_path,
                       arcname=os.path.basename(audio_path))

        open(done_marker, "w").close()

    zipf.writestr(
        "config.json",
        json.dumps({"language": language_hint.lower()},
                   indent=4, ensure_ascii=False)
    )

    zipf.close()

    print(f"Created archive: {zip_name}")
