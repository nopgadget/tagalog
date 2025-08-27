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

def generate_audio(model, tokenizer, text, speech_rate=1.0):
    """Generate audio from text using the TTS model"""
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")
    
    # Generate audio
    with torch.no_grad():
        output = model(**inputs).waveform
    
    # Convert to numpy array
    output_numpy = output.squeeze().cpu().numpy()
    
    # Apply speech rate control through audio resampling
    if speech_rate != 1.0:
        from scipy import signal
        
        # Calculate new length based on speech rate
        original_length = len(output_numpy)
        new_length = int(original_length / speech_rate)
        
        # Resample the audio to change speed
        # This maintains pitch while changing speed
        output_numpy = signal.resample(output_numpy, new_length)
    
    return output_numpy

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate Tagalog TTS audio')
    parser.add_argument('--text', '-t', type=str, default="magandang tanghali", 
                       help='Text to convert to speech (default: "magandang tanghali")')
    parser.add_argument('--speech-rate', '-s', type=float, default=0.7,
                       help='Speech rate: 0.3-2.0 (default: 0.7, lower = slower)')
    parser.add_argument('--output-dir', '-o', type=str, default="output",
                       help='Output directory (default: "output")')
    
    args = parser.parse_args()
    
    # Validate speech rate
    if args.speech_rate < 0.3 or args.speech_rate > 2.0:
        print(f"Warning: Speech rate {args.speech_rate} is out of range (0.3-2.0). Using 0.7 instead.")
        args.speech_rate = 0.7
    
    # Load model
    model, tokenizer = load_model()
    
    # Generate audio
    print(f"Generating audio for: '{args.text}'")
    print(f"Speech rate: {args.speech_rate:.1f}x")
    
    output = generate_audio(model, tokenizer, args.text, args.speech_rate)
    
    import scipy
    
    # Create output folder if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
    filename = f"tagalog_tts_{timestamp}_speed{args.speech_rate:.1f}.wav"
    output_path = os.path.join(args.output_dir, filename)
    
    # Convert PyTorch tensor to numpy array and squeeze to remove extra dimensions
    output_numpy = output.squeeze().cpu().numpy()
    
    scipy.io.wavfile.write(output_path, rate=model.config.sampling_rate, data=output_numpy)
    print(f"Audio saved to: {output_path}")

if __name__ == "__main__":
    main()