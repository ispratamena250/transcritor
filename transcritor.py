import whisper
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Uso: python transcritor.py <arquivo-de-audio>")
        return

    caminho_audio = sys.argv[1]

    if not os.path.exists(caminho_audio):
        print(f"Erro: arquivo '{caminho_audio}' não encontrado.")
        return

    print("Carregando modelo Whisper...")
    model = whisper.load_model("small", device="cpu")

    print(f"Transcrevendo: {caminho_audio}")

    resultado = model.transcribe(
        caminho_audio,
        language="pt",
        fp16=False  # Necessário na CPU
    )

    saida = "transcricao.dat"
    with open(saida, "w", encoding="utf-8") as f:
        f.write(resultado["text"])

    print(f"Transcrição salva em {saida}")

if __name__ == "__main__":
    main()