import torch
import numpy as np
from transformers import AutoProcessor, BarkModel
from scipy.io.wavfile import write
from IPython.display import Audio, display
import os

class bark_model:
    def __init__(self, model_dir="bark_saved"):
        self.model_dir = model_dir

        # Check if a saved model exists
        if os.path.exists(model_dir):
            print("📦 Loading Bark model from local directory...")
            self.processor = AutoProcessor.from_pretrained(model_dir)
            self.model = BarkModel.from_pretrained(model_dir)
        else:
            print("⬇️ Downloading Bark model from Hugging Face...")
            self.processor = AutoProcessor.from_pretrained("suno/bark")
            self.model = BarkModel.from_pretrained("suno/bark")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def generate_audio(self, text, voice_preset="v2/en_speaker_6"):
        """Generate waveform from text using Bark"""
        inputs = self.processor(text, voice_preset=voice_preset, return_tensors="pt").to(self.device)

        with torch.no_grad():
            audio_array = self.model.generate(**inputs)

        audio_array = audio_array[0].cpu().numpy()
        sample_rate = self.model.generation_config.sample_rate
        return audio_array, sample_rate

    def save_audio(self, audio_array, sample_rate, file_name="output.wav"):
        """Save the generated audio"""
        audio_int16 = np.int16(audio_array / np.max(np.abs(audio_array)) * 32767)
        write(file_name, sample_rate, audio_int16)
        print(f"✅ Audio saved to {file_name}")

    def play_audio(self, audio_array, sample_rate):
        """Play the audio"""
        display(Audio(audio_array, rate=sample_rate))
