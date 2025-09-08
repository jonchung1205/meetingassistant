# Aura Note

# Enhanced Real-Time Meeting Assistant

A powerful real-time meeting transcription and speaker identification system built with Python (FastAPI backend) and JavaScript (frontend). The system provides live transcription, multi-speaker detection, conversation analytics, and real-time meeting insights.

## Features

- **Real-Time Transcription**: Live speech-to-text using Whisper AI
- **Multi-Speaker Detection**: Advanced voice embeddings for speaker identification
- **Voice Activity Detection**: Multiple VAD methods for accurate speech detection
- **Live Analytics**: Real-time speaking time, word count, and participation metrics
- **WebSocket Communication**: Live updates without page refreshes
- **Conversation Insights**: Detailed meeting analytics and statistics
- **Robust Audio Processing**: Enhanced noise handling and audio preprocessing

## Tech Stack

### Backend
- **FastAPI**: Modern Python web framework
- **Whisper**: OpenAI's speech recognition model
- **librosa**: Audio feature extraction
- **sounddevice**: Real-time audio capture
- **webrtcvad**: Voice activity detection
- **numpy/scipy**: Scientific computing
- **asyncio**: Asynchronous processing

### Frontend
- **Vanilla JavaScript**: No framework dependencies
- **WebSocket API**: Real-time communication
- **Responsive Design**: Works on desktop and mobile
- **Chart.js**: Data visualization for meeting analytics

## Prerequisites

- Python 3.8 or higher
- Node.js (optional, for development server)
- Microphone access
- Modern web browser with WebSocket support

## Installation

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/meeting-assistant.git
   cd meeting-assistant
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. **Create environment file** (optional)
   ```bash
   cp .env.example .env
   # Edit .env with your configurations if needed
   ```

5. **Run the backend server**
   ```bash
   python server.py
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn server:app --host 0.0.0.0 --port 8001 --reload
   ```

   The backend API will be available at `http://localhost:8001`

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Serve the frontend**
   
   **Option A: Using Python's built-in server**
   ```bash
   python -m http.server 8000
   ```
   
   **Option B: Using Node.js (if installed)**
   ```bash
   npx serve -s . -p 8000
   ```
   
   **Option C: Using any static file server**
   ```bash
   # Example with nginx, apache, or any static server
   # Point document root to the frontend directory
   ```

3. **Access the application**
   Open your browser and navigate to `http://localhost:8000`

## Usage

### Starting a Meeting

1. Open the web application in your browser
2. Grant microphone permissions when prompted
3. Click "Start Meeting" to begin recording
4. The system will automatically detect speakers and transcribe speech
5. View real-time statistics and transcriptions on the dashboard

### During the Meeting

- **Live Transcription**: See transcriptions appear in real-time
- **Speaker Detection**: Different speakers are automatically identified
- **Speaking Time**: Monitor who's talking and for how long
- **Voice Activity**: Visual indicators show when speech is detected

### After the Meeting

1. Click "Stop Meeting" to end the session
2. View final meeting summary and analytics
3. Export conversation logs (if implemented)

## API Documentation

### Health Check
```bash
GET /api/health
```

### Meeting Control
```bash
POST /api/meeting/start    # Start recording session
POST /api/meeting/stop     # Stop recording session
GET /api/meeting/status    # Get current meeting status
```

### Analytics
```bash
GET /api/meeting/conversation-insights    # Get detailed analytics
GET /api/debug/audio                     # Debug audio system status
```

### WebSocket Connection
```bash
WS /api/ws    # Real-time communication endpoint
```

## Configuration

### Audio Settings
- **Sample Rate**: 16kHz (configurable in `server.py`)
- **Channels**: Mono (1 channel)
- **Frame Size**: 50ms processing windows
- **VAD Sensitivity**: Multiple aggressiveness levels

### Speaker Detection
- **Similarity Threshold**: 0.45 (adjustable for speaker sensitivity)
- **Max Speakers**: 10 concurrent speakers
- **Feature Vector**: 200+ dimensional voice embeddings

### Transcription
- **Model**: Whisper base.en
- **Language**: English (configurable)
- **Quality Filtering**: Automatic low-quality segment removal

## Development

### Backend Development

1. **Enable debug logging**
   ```python
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Hot reload during development**
   ```bash
   uvicorn server:app --reload --host 0.0.0.0 --port 8001
   ```

3. **Testing endpoints**
   ```bash
   curl http://localhost:8001/api/health
   ```

### Frontend Development

1. **Auto-reload with live server**
   ```bash
   # Using VS Code Live Server extension
   # Or any development server with auto-reload
   ```

2. **WebSocket testing**
   ```javascript
   // Test WebSocket connection in browser console
   const ws = new WebSocket('ws://localhost:8001/api/ws');
   ws.onmessage = (event) => console.log(JSON.parse(event.data));
   ```

## Troubleshooting

### Common Issues

**Audio not detected**
- Check microphone permissions in browser
- Verify microphone is working in other applications
- Check browser console for errors

**No speakers detected**
- Speak louder or closer to microphone
- Adjust similarity threshold in configuration
- Check VAD sensitivity settings

**WebSocket connection fails**
- Ensure backend is running on correct port
- Check firewall/proxy settings
- Verify WebSocket URL in frontend

**Transcription quality issues**
- Reduce background noise
- Ensure clear speech
- Check Whisper model installation

### Debug Endpoints

Use the debug endpoint to diagnose issues:
```bash
curl http://localhost:8001/api/debug/audio
```

## File Structure

```
meeting-assistant/
├── backend/
│   ├── server.py           # Main FastAPI application
│   ├── requirements.txt    # Python dependencies
│   └── .env.example       # Environment variables template
├── frontend/
│   ├── index.html         # Main HTML file
│   ├── app.js            # Frontend JavaScript
│   ├── styles.css        # Styling
│   └── assets/           # Static assets
└── README.md             # This file
```

## Performance Notes

- **CPU Usage**: Moderate during active transcription
- **Memory**: ~200-500MB depending on meeting length
- **Latency**: ~200-500ms for transcription
- **Browser Support**: Chrome, Firefox, Safari, Edge (latest versions)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Acknowledgments

- OpenAI Whisper for speech recognition
- FastAPI for the excellent web framework
- The open-source audio processing community
