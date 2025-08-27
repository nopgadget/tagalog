from flask import Flask, render_template, request, jsonify, send_file
import os
from datetime import datetime
from transformers import VitsModel, AutoTokenizer
import torch
import scipy.io.wavfile
import tempfile
import uuid

app = Flask(__name__)

# Global variables to store the model and tokenizer
model = None
tokenizer = None

def load_model():
    """Load the TTS model and tokenizer"""
    global model, tokenizer
    if model is None:
        print("Loading TTS model...")
        model = VitsModel.from_pretrained("facebook/mms-tts-tgl")
        tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-tgl")
        print("Model loaded successfully!")

def generate_audio(text):
    """Generate audio from text using the TTS model"""
    if model is None or tokenizer is None:
        load_model()
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")
    
    # Generate audio
    with torch.no_grad():
        output = model(**inputs).waveform
    
    # Convert to numpy array
    output_numpy = output.squeeze().cpu().numpy()
    
    # Create a temporary file with unique name
    temp_filename = f"temp_audio_{uuid.uuid4().hex}.wav"
    temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
    
    # Save audio to temporary file
    output_numpy = output_numpy.astype('float32')  # Ensure proper data type
    scipy.io.wavfile.write(temp_path, rate=model.config.sampling_rate, data=output_numpy)
    
    return temp_path

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/generate_tts', methods=['POST'])
def generate_tts():
    """Generate TTS audio from text"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Please provide text input'}), 400
        
        # Generate audio
        audio_path = generate_audio(text)
        
        # Return the temporary filename for the client to request
        return jsonify({
            'success': True,
            'audio_file': os.path.basename(audio_path),
            'message': f'Generated audio for: "{text}"'
        })
        
    except Exception as e:
        return jsonify({'error': f'Error generating audio: {str(e)}'}), 500

@app.route('/audio/<filename>')
def serve_audio(filename):
    """Serve the generated audio file"""
    try:
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        if os.path.exists(temp_path):
            return send_file(temp_path, mimetype='audio/wav')
        else:
            return jsonify({'error': 'Audio file not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Error serving audio: {str(e)}'}), 500

@app.route('/cleanup/<filename>')
def cleanup_audio(filename):
    """Clean up temporary audio files"""
    try:
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        if os.path.exists(temp_path):
            os.remove(temp_path)
            return jsonify({'success': True, 'message': 'Audio file cleaned up'})
        else:
            return jsonify({'error': 'Audio file not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Error cleaning up: {str(e)}'}), 500

if __name__ == '__main__':
    # Load the model when starting the app
    load_model()
    print("Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5000)
