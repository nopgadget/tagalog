from transformers import VitsModel, AutoTokenizer
import torch
import os
from datetime import datetime

model = VitsModel.from_pretrained("facebook/mms-tts-tgl")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-tgl")

text = "magandang tanghali"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform

import scipy

# Create output folder if it doesn't exist
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# Generate timestamp-based filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
filename = f"tagalog_tts_{timestamp}.wav"
output_path = os.path.join(output_folder, filename)

# Convert PyTorch tensor to numpy array and squeeze to remove extra dimensions
output_numpy = output.squeeze().cpu().numpy()

scipy.io.wavfile.write(output_path, rate=model.config.sampling_rate, data=output_numpy)
print(f"Audio saved to: {output_path}")