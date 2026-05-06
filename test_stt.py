from kokoro_onnx import Kokoro
import soundfile as sf

print("Chargement Kokoro...")
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
print("Kokoro chargé ✅")

# Génère un audio de test en français
samples, sample_rate = kokoro.create(
    "Bonjour, je suis Alex, votre assistant virtuel.",
    voice="ff_siwis",   # voix française féminine
    speed=1.0,
    lang="fr-fr"
)

sf.write("test_output.wav", samples, sample_rate)
print("Audio généré → test_output.wav ✅")