import whisper
import sys
import os

def main():
    if len(sys.argv) < 3:
        print("Error. Run: python transcritor.py <audio-file> <output-file>")
        return

    audio_path = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.exists(audio_path):
        print(f"Error: file '{audio_path}' not found.")
        return

    output_dir = os.path.exists(audio_path)
    if output_dir != "" and not os.path.exists(output_dir):
        print(f"Error: directory {output_dir} does not exist")
        return

    print("Loading Whisper model...")
    model = whisper.load_model("small", device="cpu")

    print(f"Transcribing: {audio_path}")

    resultado = model.transcribe(
        audio_path,
        language="pt",
        fp16=False
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(resultado["text"])

    print(f"Transcription saved in {output_file}")

if __name__ == "__main__":
    main()