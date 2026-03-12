import json
import os
import zipfile

SUPPORTED_EXTENSIONS = [
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
]


def run_stage3(text_dir, audio_dir, zip_name, language):
    zip_folder = os.path.dirname(zip_name)
    if zip_folder and not os.path.exists(zip_folder):
        os.makedirs(zip_folder)

    txt_files = [f for f in os.listdir(text_dir) if f.endswith(".txt")]
    print(f"Creating Zip: {zip_name}")

    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zout:
        zout.writestr("config.json", json.dumps({"language": language}, indent=4))

        for txt_file in txt_files:
            print(f"Packaging: {txt_file}")
            txt_path = os.path.join(text_dir, txt_file)
            base_name = os.path.splitext(txt_file)[0]

            media_path = None
            found_media_name = None
            for ext in SUPPORTED_EXTENSIONS:
                cand = os.path.join(audio_dir, base_name + ext)
                if os.path.exists(cand):
                    media_path = cand
                    found_media_name = base_name + ext
                    break

            if media_path:
                # Add source media to zip
                zout.write(media_path, arcname=found_media_name)
                # Add the finalized text transcript to zip
                zout.write(txt_path, arcname=txt_file)
            else:
                print(f" - Warning: No source media found for {txt_file}")

    print("Stage 3 Complete: Zip created successfully.")
