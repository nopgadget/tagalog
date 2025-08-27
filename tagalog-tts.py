from transformers import VitsModel, AutoTokenizer
import torch
import os
from datetime import datetime
import argparse

def load_model():
    """Load the TTS model and tokenizer"""
    print("Loading TTS model...")
    model = VitsModel.from_pretrained("facebook/mms-tts-tgl")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-tgl")
    print("Model loaded successfully!")
    return model, tokenizer

def generate_audio(model, tokenizer, text):
    """Generate audio from text using the TTS model"""
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")
    
    # Generate audio
    with torch.no_grad():
        output = model(**inputs).waveform
    
    # Convert to numpy array
    output_numpy = output.squeeze().cpu().numpy()
    
    return output_numpy

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate Tagalog TTS audio')
    parser.add_argument('--text', '-t', type=str, default="magandang tanghali", 
                       help='Text to convert to speech (default: "magandang tanghali")')
    parser.add_argument('--output-dir', '-o', type=str, default="output",
                       help='Output directory (default: "output")')
    
    args = parser.parse_args()
    

    
    # Load model
    model, tokenizer = load_model()
    
    # Generate audio
    print(f"Generating audio for: '{args.text}'")
    
    output = generate_audio(model, tokenizer, args.text)
    
    import scipy
    
    # Create output folder if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
    filename = f"tagalog_tts_{timestamp}.wav"
    output_path = os.path.join(args.output_dir, filename)
    
    scipy.io.wavfile.write(output_path, rate=model.config.sampling_rate, data=output)
    print(f"Audio saved to: {output_path}")

if __name__ == "__main__":
    main()