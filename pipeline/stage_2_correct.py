import gc
import os

try:
    from llama_cpp import Llama
except ImportError:
    print("Error: Missing llama-cpp-python. Check requirements.txt!")
    raise


def is_obvious_garbage(sentence):
    s = sentence.strip()
    return not s or len(s) < 2


def is_valid_sentence_llm(llm, sentence, language_hint):
    prompt = f'Determine if the following text is spoken in {language_hint}. Reply strictly with YES or NO.\nText: "{sentence}"\nAnswer:'
    try:
        output = llm(prompt, max_tokens=5, stop=["\n", ".", ","], echo=False)
        response = output["choices"][0]["text"].strip().upper()

        # If the LLM explicitly says "NO", it's English instructions. We flag it for deletion.
        if "NO" in response and "YES" not in response:
            return False

        return True
    except Exception as e:
        print(f"LLM Error on text '{sentence}': {e}")
        return True


def run_stage2(
    language_hint, transcribed_dir, corrected_dir, llm_model_path, force_cpu=False
):
    if not os.path.exists(corrected_dir):
        os.makedirs(corrected_dir)

    print(f"Loading LLM from {llm_model_path}...")
    n_gpu_layers = 0 if force_cpu else -1
    llm = Llama(
        model_path=llm_model_path, n_ctx=2048, n_gpu_layers=n_gpu_layers, verbose=False
    )

    txt_files = [f for f in os.listdir(transcribed_dir) if f.endswith(".txt")]
    for txt_file in txt_files:
        print(f"Filtering English from: {txt_file}")
        txt_path = os.path.join(transcribed_dir, txt_file)
        out_path = os.path.join(corrected_dir, txt_file)

        with open(txt_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        cleaned_lines = []

        print(f" -> Evaluating {len(lines)} lines... (This may take a moment)")
        for line in lines:
            if " " in line:
                timestamp, text = line.split(" ", 1)
            else:
                timestamp, text = "00:00", line

            text = text.strip()
            if is_obvious_garbage(text):
                continue

            is_target_language = is_valid_sentence_llm(llm, text, language_hint)

            # If it IS the target language, we keep it! If it's English, we completely drop it.
            if is_target_language:
                cleaned_lines.append(f"{timestamp} {text}")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(cleaned_lines))

    print("\n[System] Unloading LLM from memory...")
    del llm
    gc.collect()

    print(f"Stage 2 Complete. Cleaned files saved to {corrected_dir}")
