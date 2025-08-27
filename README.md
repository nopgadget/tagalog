# Tagalog Text-to-Speech (TTS) System

A Python-based Text-to-Speech system for the Tagalog language using Facebook's MMS-TTS-TGL model.

## Features

- 🎤 High-quality Tagalog speech synthesis
- 🌐 Web interface for easy use
- 💻 Command-line interface for batch processing
- 📁 Automatic file organization with timestamped outputs
- 🎵 Real-time audio generation and playback

## Installation

1. **Clone or download this repository**
2. **Install dependencies:**
   ```bash
   conda install numpy scipy pytorch transformers flask
   ```

## Usage

### 🌐 Web Interface (Recommended)

1. **Start the web server:**
   ```bash
   python tagalog-tts-web.py
   ```

2. **Open your browser and go to:**
   ```
   http://localhost:5000
   ```

3. **Features:**
   - Enter Tagalog text in the text area
   - Click "Generate Speech" to create audio
   - Use example buttons for quick testing
   - Audio automatically plays after generation
   - Responsive design works on mobile and desktop

### 💻 Command Line Interface

1. **Basic usage:**
   ```bash
   python tagalog-tts.py
   ```

2. **Customize the text:**
   - Edit the `text` variable in `tagalog-tts.py`
   - Run the script to generate audio

3. **Output:**
   - Creates an `output/` folder automatically
   - Generates timestamped WAV files (e.g., `tagalog_tts_20241220_143052_123.wav`)

## Example Tagalog Phrases

- **Magandang umaga!** - Good morning!
- **Kumusta ka?** - How are you?
- **Salamat po** - Thank you (polite)
- **Mahal kita** - I love you
- **Paalam na po** - Goodbye (polite)

## Technical Details

- **Model:** Facebook MMS-TTS-TGL (VitsModel)
- **Audio Format:** WAV (16-bit PCM)
- **Sampling Rate:** Model-dependent (typically 22.05 kHz)
- **Framework:** PyTorch + Transformers
- **Web Framework:** Flask

## File Structure

```
tagalog/
├── tagalog-tts.py          # Command-line TTS script
├── tagalog-tts-web.py      # Flask web application
├── templates/
│   └── index.html          # Web interface template
├── output/                 # Generated audio files (created automatically)
└── README.md               # This file
```

## Troubleshooting

### Common Issues

1. **Import errors:**
   - Ensure all dependencies are installed
   - Check that you're using the correct conda environment

2. **Model loading issues:**
   - The model will download automatically on first run
   - Requires internet connection for initial download

3. **Audio playback issues:**
   - Check browser audio settings
   - Ensure audio files are generated successfully

### Performance Tips

- The model loads once when starting the web server
- Subsequent requests are faster as the model stays in memory
- Audio generation typically takes 2-5 seconds depending on text length

## License

This project uses the Facebook MMS-TTS-TGL model. Please refer to the model's license for usage terms.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this TTS system!
