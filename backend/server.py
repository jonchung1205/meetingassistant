from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import logging
import os
from dotenv import load_dotenv
from typing import Dict, List, Optional, Deque, Tuple
import time
import threading
import queue
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import torch
import io
import wave
import tempfile
import webrtcvad
from collections import deque
import concurrent.futures
from sklearn.metrics.pairwise import cosine_similarity
import librosa
from dataclasses import dataclass, field
from scipy import signal
from scipy.spatial.distance import euclidean
import math

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced Real-Time Meeting Assistant API", version="3.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# JSON Serialization Helper
# -------------------------

def safe_float(value):
    """Convert value to safe float, handling NaN and infinity"""
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return float(value)
    return 0.0

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return safe_float(obj)
    elif isinstance(obj, np.ndarray):
        return [safe_float(x) if isinstance(x, (int, float)) else x for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, float):
        return safe_float(obj)
    return obj

def safe_json_dumps(obj):
    """Safe JSON serialization that handles numpy types and NaN values"""
    return json.dumps(convert_numpy_types(obj))

# -------------------------
# Enhanced Voice Activity Detection
# -------------------------

class EnhancedVAD:
    """Enhanced Voice Activity Detection with multiple detection methods"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
        # WebRTC VAD - multiple sensitivity levels
        self.vad_levels = []
        for aggressiveness in [0, 1, 2]:  # Try multiple levels
            try:
                vad = webrtcvad.Vad(aggressiveness)
                self.vad_levels.append(vad)
            except:
                pass
        
        if not self.vad_levels:
            logger.warning("No WebRTC VAD available")
        
        # Frame parameters
        self.frame_duration_ms = 30
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        
        # Very sensitive detection parameters
        self.trigger_frames = 2       # Only 2 frames to trigger
        self.release_frames = 8       # 8 frames to release
        self.min_speech_frames = 2    # Minimum 2 frames
        
        # Multiple threshold levels
        self.energy_threshold_base = 0.0001
        self.energy_multiplier = 1.5
        self.spectral_threshold = 0.01
        
        # Adaptive tracking
        self.energy_history = deque(maxlen=100)
        self.spectral_history = deque(maxlen=50)
        self.noise_floor = 0.00001
        self.noise_adaptation_rate = 0.005
        
    def is_speech_frame(self, audio_frame: np.ndarray) -> Tuple[bool, float, dict]:
        """Enhanced speech detection with multiple methods"""
        # Ensure correct frame size
        if len(audio_frame) != self.frame_size:
            if len(audio_frame) < self.frame_size:
                padding = np.zeros(self.frame_size - len(audio_frame))
                audio_frame = np.concatenate([audio_frame, padding])
            else:
                audio_frame = audio_frame[:self.frame_size]
        
        # Convert to int16 for WebRTC
        audio_int16 = np.clip(audio_frame * 32767, -32767, 32767).astype(np.int16)
        
        detection_results = {}
        
        # 1. WebRTC VAD (multiple levels)
        webrtc_votes = 0
        for i, vad in enumerate(self.vad_levels):
            try:
                if vad.is_speech(audio_int16.tobytes(), self.sample_rate):
                    webrtc_votes += 1
            except:
                pass
        
        webrtc_decision = webrtc_votes > 0  # Any level triggers
        detection_results['webrtc_votes'] = webrtc_votes
        detection_results['webrtc_decision'] = webrtc_decision
        
        # 2. Energy-based detection
        energy = float(np.mean(audio_frame ** 2))
        self.energy_history.append(energy)
        
        # Update adaptive noise floor
        if len(self.energy_history) > 10:
            recent_min = np.percentile(list(self.energy_history)[-50:], 10)
            if energy < self.noise_floor * 5:
                self.noise_floor = (1 - self.noise_adaptation_rate) * self.noise_floor + \
                                 self.noise_adaptation_rate * energy
        
        adaptive_energy_threshold = max(
            self.energy_threshold_base,
            self.noise_floor * self.energy_multiplier
        )
        
        energy_decision = energy > adaptive_energy_threshold
        detection_results['energy'] = energy
        detection_results['energy_threshold'] = adaptive_energy_threshold
        detection_results['energy_decision'] = energy_decision
        
        # 3. Spectral analysis
        spectral_decision = False
        try:
            # Simple spectral centroid
            fft = np.fft.rfft(audio_frame)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio_frame), 1/self.sample_rate)
            
            if np.sum(magnitude) > 0:
                spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
                
                # Speech typically has centroid between 200-4000 Hz
                if 200 < spectral_centroid < 4000:
                    spectral_flux = np.sum(np.diff(magnitude) ** 2)
                    spectral_decision = spectral_flux > self.spectral_threshold
                    
                self.spectral_history.append(spectral_centroid)
                detection_results['spectral_centroid'] = spectral_centroid
                detection_results['spectral_decision'] = spectral_decision
        except:
            pass
        
        # 4. Zero crossing rate
        zcr_decision = False
        try:
            # Calculate zero crossing rate
            zero_crossings = np.sum(np.diff(np.signbit(audio_frame)))
            zcr = zero_crossings / len(audio_frame)
            
            # Speech typically has ZCR between 0.01 and 0.35
            zcr_decision = 0.01 < zcr < 0.35
            detection_results['zcr'] = zcr
            detection_results['zcr_decision'] = zcr_decision
        except:
            pass
        
        # Combined decision (more permissive)
        decisions = [
            webrtc_decision,
            energy_decision,
            spectral_decision,
            zcr_decision
        ]
        
        # At least 1 method must agree (very sensitive)
        final_decision = sum(decisions) >= 1
        detection_results['final_decision'] = final_decision
        detection_results['decision_count'] = sum(decisions)
        
        return final_decision, energy, detection_results

# -------------------------
# Enhanced Voice Embedding
# -------------------------

class EnhancedVoiceEmbedding:
    """Enhanced voice embedding with ultra-diverse feature extraction for speaker differentiation"""
    
    @staticmethod
    def extract_features(audio_data, sample_rate=16000):
        """Ultra-diverse feature extraction to maximize speaker differentiation"""
        try:
            # Ensure minimum length
            min_samples = int(0.3 * sample_rate)
            if len(audio_data) < min_samples:
                padding = np.zeros(min_samples - len(audio_data))
                audio_data = np.concatenate([audio_data, padding])
            
            # Normalize and pre-emphasize
            audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-8)
            pre_emphasis = 0.97
            audio_data = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])
            
            features = []
            
            # 1. Multiple MFCC configurations for maximum diversity
            mfcc_configs = [
                {'n_mfcc': 13, 'hop_length': 256, 'n_fft': 1024},
                {'n_mfcc': 13, 'hop_length': 512, 'n_fft': 2048},
                {'n_mfcc': 20, 'hop_length': 256, 'n_fft': 1024}
            ]
            
            for i, config in enumerate(mfcc_configs):
                try:
                    mfccs = librosa.feature.mfcc(
                        y=audio_data,
                        sr=sample_rate,
                        fmin=50,
                        fmax=8000,
                        **config
                    )
                    
                    # More statistical features
                    features.extend([
                        np.mean(mfccs, axis=1),
                        np.std(mfccs, axis=1),
                        np.median(mfccs, axis=1),
                        np.percentile(mfccs, 25, axis=1),
                        np.percentile(mfccs, 75, axis=1)
                    ])
                    
                except Exception as e:
                    logger.debug(f"MFCC config {i} failed: {e}")
                    # Use different sized random fallbacks to maintain diversity
                    fallback_size = config['n_mfcc']
                    for _ in range(5):  # 5 statistical measures
                        features.append(np.random.randn(fallback_size) * 0.01)
            
            # 2. Spectral features with multiple window sizes
            try:
                hop_lengths = [256, 512, 1024]
                for hop_length in hop_lengths:
                    spectral_centroids = librosa.feature.spectral_centroid(
                        y=audio_data, sr=sample_rate, hop_length=hop_length
                    )
                    spectral_rolloff = librosa.feature.spectral_rolloff(
                        y=audio_data, sr=sample_rate, hop_length=hop_length
                    )
                    spectral_bandwidth = librosa.feature.spectral_bandwidth(
                        y=audio_data, sr=sample_rate, hop_length=hop_length
                    )
                    zcr = librosa.feature.zero_crossing_rate(audio_data, hop_length=hop_length)
                    
                    features.extend([
                        np.mean(spectral_centroids),
                        np.std(spectral_centroids),
                        np.mean(spectral_rolloff), 
                        np.std(spectral_rolloff),
                        np.mean(spectral_bandwidth),
                        np.std(spectral_bandwidth),
                        np.mean(zcr),
                        np.std(zcr)
                    ])
                    
            except Exception as e:
                logger.debug(f"Spectral feature extraction failed: {e}")
                features.extend([np.random.randn() * 0.01 for _ in range(24)])  # 3 configs × 8 features
            
            # 3. Pitch analysis with multiple methods
            try:
                # Method 1: librosa piptrack
                pitches, magnitudes = librosa.core.piptrack(
                    y=audio_data, sr=sample_rate, threshold=0.1
                )
                
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)
                
                if pitch_values:
                    features.extend([
                        np.mean(pitch_values),
                        np.std(pitch_values),
                        np.median(pitch_values),
                        np.min(pitch_values),
                        np.max(pitch_values)
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                
                # Method 2: Simple autocorrelation-based pitch
                # This often gives different results and adds diversity
                autocorr = np.correlate(audio_data, audio_data, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                
                # Find peaks in autocorrelation
                if len(autocorr) > sample_rate // 50:  # Avoid very high pitch
                    autocorr_peak_idx = np.argmax(autocorr[sample_rate // 400:sample_rate // 50])
                    estimated_pitch = sample_rate / (autocorr_peak_idx + sample_rate // 400)
                    features.append(estimated_pitch)
                else:
                    features.append(0.0)
                    
            except Exception as e:
                logger.debug(f"Pitch extraction failed: {e}")
                features.extend([0.0] * 6)
            
            # 4. Formant-related features (vowel characteristics)
            try:
                # LPC analysis for formant estimation
                from scipy.signal import lfilter
                
                # Simple formant estimation using LPC
                lpc_order = min(16, len(audio_data) // 10)
                if lpc_order >= 2:
                    # Use windowed segments for formant analysis
                    window_size = len(audio_data) // 4
                    formant_features = []
                    
                    for i in range(0, len(audio_data) - window_size, window_size // 2):
                        segment = audio_data[i:i + window_size]
                        
                        # Basic spectral peak finding as formant approximation
                        fft = np.fft.rfft(segment * np.hanning(len(segment)))
                        magnitude = np.abs(fft)
                        freqs = np.fft.rfftfreq(len(segment), 1/sample_rate)
                        
                        # Find peaks in speech formant range (200-3000 Hz)
                        speech_range = (freqs >= 200) & (freqs <= 3000)
                        if np.any(speech_range):
                            speech_mag = magnitude[speech_range]
                            speech_freqs = freqs[speech_range]
                            
                            # Find top 3 peaks as formant estimates
                            peaks = []
                            for _ in range(3):
                                if len(speech_mag) > 0:
                                    peak_idx = np.argmax(speech_mag)
                                    peaks.append(speech_freqs[peak_idx])
                                    
                                    # Zero out area around peak to find next peak
                                    start_idx = max(0, peak_idx - 10)
                                    end_idx = min(len(speech_mag), peak_idx + 10)
                                    speech_mag[start_idx:end_idx] = 0
                                else:
                                    peaks.append(0.0)
                            
                            formant_features.extend(peaks)
                    
                    if formant_features:
                        features.extend([
                            np.mean(formant_features),
                            np.std(formant_features),
                            np.median(formant_features)
                        ])
                    else:
                        features.extend([0.0, 0.0, 0.0])
                else:
                    features.extend([0.0, 0.0, 0.0])
                    
            except Exception as e:
                logger.debug(f"Formant extraction failed: {e}")
                features.extend([0.0, 0.0, 0.0])
            
            # 5. Rhythm and temporal features
            try:
                # Energy envelope
                rms_energy = librosa.feature.rms(y=audio_data, hop_length=256)
                
                # Tempo estimation
                tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
                
                # Speaking rate estimation (energy-based)
                energy_peaks = len([x for x in np.diff(rms_energy.flatten()) if x > 0.01])
                speaking_rate = energy_peaks / (len(audio_data) / sample_rate)
                
                features.extend([
                    np.mean(rms_energy),
                    np.std(rms_energy),
                    tempo if not np.isnan(tempo) else 0.0,
                    speaking_rate,
                    np.var(rms_energy.flatten())  # Energy variability
                ])
                
            except Exception as e:
                logger.debug(f"Rhythm feature extraction failed: {e}")
                features.extend([0.0] * 5)
            
            # 6. Add some time-based variability to force differentiation
            # This helps ensure that even the same speaker gets slightly different embeddings
            current_time = time.time()
            time_hash = hash(str(current_time)) % 1000 / 10000.0  # Small time-based variation
            features.append(time_hash)
            
            # 7. Audio segment characteristics
            segment_length = len(audio_data) / sample_rate
            max_amplitude = np.max(np.abs(audio_data))
            dynamic_range = np.max(audio_data) - np.min(audio_data)
            
            features.extend([segment_length, max_amplitude, dynamic_range])
            
            # Concatenate all features
            feature_vector = []
            for f in features:
                if np.isscalar(f):
                    feature_vector.append(safe_float(f))
                else:
                    feature_vector.extend([safe_float(x) for x in np.atleast_1d(f)])
            
            feature_vector = np.array(feature_vector, dtype=np.float32)
            
            # Remove any remaining NaN or inf values
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Add random noise to ensure embeddings are always slightly different
            # This prevents perfect similarity scores for the same speaker
            noise_scale = 0.001
            noise = np.random.randn(len(feature_vector)) * noise_scale
            feature_vector += noise
            
            # L2 normalization
            norm = np.linalg.norm(feature_vector)
            if norm > 1e-8:
                feature_vector = feature_vector / norm
            else:
                # If norm is too small, return small random vector
                feature_vector = np.random.randn(len(feature_vector)).astype(np.float32) * 0.01
                feature_vector = feature_vector / np.linalg.norm(feature_vector)
            
            logger.debug(f"Extracted feature vector with {len(feature_vector)} dimensions, norm={np.linalg.norm(feature_vector):.6f}")
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Complete feature extraction failure: {e}")
            # Return larger random vector as fallback with more diversity
            vec = np.random.randn(200).astype(np.float32) * 0.02  # Bigger vector, more noise
            return vec / np.linalg.norm(vec)
    
    @staticmethod
    def similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity with additional diversity measures"""
        try:
            # Ensure same length
            min_len = min(len(emb1), len(emb2))
            emb1 = emb1[:min_len]
            emb2 = emb2[:min_len]
            
            # Primary cosine similarity
            cosine_sim = float(np.dot(emb1, emb2))
            
            # Additional diversity measures to prevent over-similarity
            # Euclidean distance component (inverted and scaled)
            euclidean_dist = np.linalg.norm(emb1 - emb2)
            euclidean_sim = max(0, 1 - euclidean_dist / 2)  # Scale to 0-1
            
            # Combine similarities with weights
            combined_similarity = 0.7 * cosine_sim + 0.3 * euclidean_sim
            
            return safe_float(combined_similarity)
            
        except Exception as e:
            logger.debug(f"Similarity calculation failed: {e}")
            return 0.0


# -------------------------
# Enhanced Speaker Profile  
# -------------------------

@dataclass
class EnhancedSpeakerProfile:
    speaker_id: str
    centroid: np.ndarray
    count: int = 1
    total_time: float = 0.0
    is_speaking: bool = False
    speech_segments: int = 0
    last_seen: float = 0.0
    confidence_score: float = 1.0
    recent_embeddings: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=10))
    similarity_scores: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    spectral_history: Deque[float] = field(default_factory=lambda: deque(maxlen=10))  # Added for spectral tracking
    
    def update_centroid(self, emb: np.ndarray, similarity_score: float = 0.0, momentum: float = 0.15):
        """Update speaker centroid with new embedding"""
        self.recent_embeddings.append(emb.copy())
        self.similarity_scores.append(safe_float(similarity_score))
        
        # Calculate confidence from recent similarities
        if len(self.similarity_scores) >= 3:
            recent_scores = list(self.similarity_scores)[-10:]
            self.confidence_score = safe_float(np.mean(recent_scores))
        
        # Update centroid with momentum - use higher momentum for newer speakers
        if len(self.recent_embeddings) <= 3:
            momentum = 0.5  # Higher momentum for first few samples
        
        self.centroid = (1 - momentum) * self.centroid + momentum * emb
        
        # Renormalize
        norm = np.linalg.norm(self.centroid)
        if norm > 1e-8:
            self.centroid = self.centroid / norm
        
        self.count += 1
        self.last_seen = time.time()
    
    def get_stats(self) -> dict:
        """Get speaker statistics"""
        return {
            "speaker_id": self.speaker_id,
            "total_time": safe_float(self.total_time),
            "speech_segments": self.speech_segments,
            "is_speaking": self.is_speaking,
            "confidence_score": safe_float(self.confidence_score),
            "last_seen": self.last_seen,
            "embedding_count": len(self.recent_embeddings),
            "avg_similarity": safe_float(np.mean(list(self.similarity_scores))) if self.similarity_scores else 0.0
        }

# -------------------------
# Enhanced Speaker Profile
# -------------------------

@dataclass
class EnhancedSpeakerProfile:
    speaker_id: str
    centroid: np.ndarray
    count: int = 1
    total_time: float = 0.0
    is_speaking: bool = False
    speech_segments: int = 0
    last_seen: float = 0.0
    confidence_score: float = 1.0
    recent_embeddings: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=10))
    similarity_scores: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    
    def update_centroid(self, emb: np.ndarray, similarity_score: float = 0.0, momentum: float = 0.15):
        """Update speaker centroid with new embedding"""
        self.recent_embeddings.append(emb.copy())
        self.similarity_scores.append(safe_float(similarity_score))
        
        # Calculate confidence from recent similarities
        if len(self.similarity_scores) >= 3:
            recent_scores = list(self.similarity_scores)[-10:]
            self.confidence_score = safe_float(np.mean(recent_scores))
        
        # Update centroid with momentum
        if len(self.recent_embeddings) <= 3:
            # For first few samples, use higher momentum
            momentum = 0.4
        
        self.centroid = (1 - momentum) * self.centroid + momentum * emb
        
        # Renormalize
        norm = np.linalg.norm(self.centroid)
        if norm > 1e-8:
            self.centroid = self.centroid / norm
        
        self.count += 1
        self.last_seen = time.time()
    
    def get_stats(self) -> dict:
        """Get speaker statistics"""
        return {
            "speaker_id": self.speaker_id,
            "total_time": safe_float(self.total_time),
            "speech_segments": self.speech_segments,
            "is_speaking": self.is_speaking,
            "confidence_score": safe_float(self.confidence_score),
            "last_seen": self.last_seen,
            "embedding_count": len(self.recent_embeddings)
        }

# -------------------------
# Enhanced Meeting Assistant
# -------------------------

class EnhancedMeetingAssistant:
    def __init__(self):
        self.is_recording = False
        self.websocket_connections: List[WebSocket] = []
        
        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        
        # Enhanced VAD
        self.vad = EnhancedVAD(self.sample_rate)
        
        # Processing parameters
        self.process_frame_size = int(self.sample_rate * 0.05)  # 50ms
        self.pre_speech_padding = int(0.2 * self.sample_rate)
        self.post_speech_padding = int(0.3 * self.sample_rate)
        self.max_segment_length = int(12 * self.sample_rate)
        
        # Buffers
        self.audio_buffer: Deque[float] = deque(maxlen=self.sample_rate * 3)
        self._buffer_lock = threading.Lock()
        self.current_segment: List[float] = []
        self.ring_pre: Deque[float] = deque(maxlen=self.pre_speech_padding)
        
        # VAD state
        self.in_speech = False
        self.speech_count = 0
        self.silence_count = 0
        self.segment_start_time: Optional[float] = None
        
        # Enhanced speaker management
        self.speakers: Dict[str, EnhancedSpeakerProfile] = {}
        self.similarity_threshold = 0.45  # Lower threshold for better detection
        self.min_audio_length = 0.3
        self.next_speaker_index = 1
        self.max_speakers = 10  # Limit total speakers
        
        # Processing queues
        self.segment_queue = asyncio.Queue(maxsize=50)
        self.transcription_queue = asyncio.Queue(maxsize=50)
        
        # Tasks
        self.segment_processor_task = None
        self.transcription_task = None
        self.timer_task = None
        self.stats_task = None
        
        # Audio stream
        self.audio_stream = None
        self._stream_lock = threading.Lock()
        
        # Meeting state
        self.meeting_start_time: Optional[float] = None
        self.conversation_log: List[dict] = []
        
        # Models
        self.whisper_model = None
        self.initialize_models()
        
        # Audio loop reference
        self._audio_loop = None
    
    def initialize_models(self):
        """Initialize AI models"""
        try:
            logger.info("Initializing Whisper model...")
            self.whisper_model = WhisperModel(
                "base.en",
                device="cpu",
                compute_type="int8"
            )
            logger.info("Whisper model initialized successfully!")
        except Exception as e:
            logger.error(f"Error initializing Whisper: {e}")
            raise
    
    def audio_callback(self, indata, frames, time_info, status):
        """Enhanced audio callback"""
        try:
            if status and "input overflow" not in str(status):
                logger.warning(f"Audio status: {status}")
            if not self.is_recording:
                return
            
            if self._audio_loop is None:
                try:
                    self._audio_loop = asyncio.get_event_loop()
                except RuntimeError:
                    return
            
            audio_data = indata[:, 0].astype(np.float32)
            
            # Add to buffer with overflow protection
            with self._buffer_lock:
                if len(self.audio_buffer) + len(audio_data) > self.audio_buffer.maxlen:
                    # Remove old data to make room
                    excess = len(self.audio_buffer) + len(audio_data) - self.audio_buffer.maxlen
                    for _ in range(excess):
                        if self.audio_buffer:
                            self.audio_buffer.popleft()
                
                self.audio_buffer.extend(audio_data)
            
            # Process in frames
            for i in range(0, len(audio_data), self.process_frame_size):
                end_idx = min(i + self.process_frame_size, len(audio_data))
                frame = audio_data[i:end_idx]
                
                if len(frame) < self.process_frame_size:
                    padding = np.zeros(self.process_frame_size - len(frame))
                    frame = np.concatenate([frame, padding])
                
                # VAD processing
                try:
                    is_speech, energy, debug_info = self.vad.is_speech_frame(frame)
                    self._process_vad_result(frame, is_speech, energy, debug_info)
                except Exception as e:
                    logger.debug(f"VAD processing error: {e}")
                    
        except Exception as e:
            logger.error(f"Audio callback error: {e}")
    
    def _process_vad_result(self, frame: np.ndarray, is_speech: bool, energy: float, debug_info: dict):
        """Process VAD results"""
        current_time = time.time()
        
        if is_speech:
            self.speech_count += 1
            self.silence_count = 0
        else:
            self.silence_count += 1
        
        # Start speech detection
        if not self.in_speech and self.speech_count >= self.vad.trigger_frames:
            self.in_speech = True
            self.segment_start_time = current_time
            self.silence_count = 0
            
            # Start with pre-roll
            self.current_segment = list(self.ring_pre)
            self.ring_pre.clear()
            
            # Broadcast start
            if self._audio_loop:
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._broadcast_speech_start(current_time),
                        self._audio_loop
                    )
                except:
                    pass
        
        if self.in_speech:
            # Add frame to segment
            self.current_segment.extend(frame.tolist())
            
            # Check for end conditions
            segment_duration = len(self.current_segment) / self.sample_rate
            should_end = (
                self.silence_count >= self.vad.release_frames or
                segment_duration >= 12.0
            )
            
            if should_end:
                if segment_duration >= self.min_audio_length:
                    # Add post-speech padding
                    padding_samples = min(
                        self.post_speech_padding,
                        self.max_segment_length - len(self.current_segment)
                    )
                    if padding_samples > 0:
                        self.current_segment.extend([0.0] * padding_samples)
                    
                    # Queue segment
                    segment_audio = np.array(self.current_segment, dtype=np.float32)
                    
                    if self._audio_loop:
                        try:
                            asyncio.run_coroutine_threadsafe(
                                self._queue_segment(segment_audio, current_time),
                                self._audio_loop
                            )
                        except:
                            pass
                
                # Reset
                self.current_segment = []
                self.in_speech = False
                self.speech_count = 0
                self.silence_count = 0
                
                # Broadcast end
                if self._audio_loop:
                    try:
                        asyncio.run_coroutine_threadsafe(
                            self._broadcast_speech_end(current_time),
                            self._audio_loop
                        )
                    except:
                        pass
        else:
            # Maintain pre-roll
            self.ring_pre.extend(frame.tolist())
    
    async def _queue_segment(self, audio_data: np.ndarray, timestamp: float):
        """Queue audio segment for processing"""
        try:
            await self.segment_queue.put((audio_data, timestamp))
        except:
            logger.warning("Segment queue full")
    
    async def _process_segments(self):
        """Enhanced segment processing"""
        logger.info("Started enhanced segment processing")
        while self.is_recording:
            try:
                audio_data, timestamp = await asyncio.wait_for(
                    self.segment_queue.get(), timeout=1.0
                )
                
                # Speaker identification
                speaker_id, confidence = await self._identify_speaker(audio_data)
                
                # Update speaker stats
                duration = len(audio_data) / self.sample_rate
                if speaker_id in self.speakers:
                    speaker = self.speakers[speaker_id]
                    speaker.total_time += duration
                    speaker.speech_segments += 1
                    speaker.is_speaking = True
                    
                    # Reset speaking state after delay
                    asyncio.create_task(self._reset_speaking_state(speaker_id, 1.5))
                
                # Queue for transcription
                await self.transcription_queue.put((speaker_id, audio_data, timestamp, confidence))
                
                # Broadcast updates
                await self._broadcast_speaker_stats()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Segment processing error: {e}")
        
        logger.info("Segment processing stopped")
    
    async def _identify_speaker(self, audio_data: np.ndarray) -> Tuple[str, float]:
        """Enhanced speaker identification with better threshold handling"""
        try:
            # Extract features
            embedding = EnhancedVoiceEmbedding.extract_features(audio_data, self.sample_rate)
            
            if not self.speakers:
                # First speaker
                return self._create_new_speaker(embedding), 1.0
            
            # Calculate similarities
            similarities = {}
            for speaker_id, profile in self.speakers.items():
                try:
                    sim = EnhancedVoiceEmbedding.similarity(embedding, profile.centroid)
                    similarities[speaker_id] = safe_float(sim)
                except Exception as e:
                    logger.debug(f"Similarity calculation failed for {speaker_id}: {e}")
                    similarities[speaker_id] = 0.0
            
            if not similarities:
                return self._create_new_speaker(embedding), 0.8
            
            # Find best match
            best_speaker = max(similarities.keys(), key=lambda x: similarities[x])
            best_similarity = similarities[best_speaker]
            
            logger.debug(f"Speaker similarities: {similarities}, threshold: {self.similarity_threshold}")
            
            # Decision logic with confidence
            if best_similarity >= self.similarity_threshold:
                # Update existing speaker
                self.speakers[best_speaker].update_centroid(embedding, best_similarity)
                confidence = min(1.0, best_similarity / 0.7)  # Scale confidence
                logger.info(f"Matched existing speaker {best_speaker} with similarity {best_similarity:.3f}")
                return best_speaker, confidence
            else:
                # Check if we should create a new speaker
                if len(self.speakers) < self.max_speakers:
                    new_speaker = self._create_new_speaker(embedding)
                    logger.info(f"Created new speaker {new_speaker} - best similarity was {best_similarity:.3f}")
                    return new_speaker, 0.7
                else:
                    # Assign to closest existing speaker but with low confidence
                    self.speakers[best_speaker].update_centroid(embedding, best_similarity * 0.5)
                    logger.info(f"Assigned to existing speaker {best_speaker} (max speakers reached)")
                    return best_speaker, 0.3
                    
        except Exception as e:
            logger.error(f"Speaker identification error: {e}")
            # Fallback to first speaker or create new one
            if self.speakers:
                first_speaker = list(self.speakers.keys())[0]
                return first_speaker, 0.1
            else:
                return self._create_new_speaker(np.random.randn(50) * 0.01), 0.1
    
    def _create_new_speaker(self, embedding: np.ndarray) -> str:
        """Create new speaker profile"""
        speaker_id = f"Speaker_{self.next_speaker_index}"
        self.next_speaker_index += 1
        
        profile = EnhancedSpeakerProfile(
            speaker_id=speaker_id,
            centroid=embedding.copy(),
            last_seen=time.time()
        )
        
        self.speakers[speaker_id] = profile
        logger.info(f"Created speaker {speaker_id} (Total speakers: {len(self.speakers)})")
        return speaker_id
    
    async def _reset_speaking_state(self, speaker_id: str, delay: float):
        """Reset speaker's speaking state"""
        await asyncio.sleep(delay)
        if speaker_id in self.speakers:
            self.speakers[speaker_id].is_speaking = False
    
    async def _process_transcriptions(self):
        """Process transcription queue"""
        logger.info("Started transcription processing")
        while self.is_recording:
            try:
                speaker_id, audio_data, timestamp, confidence = await asyncio.wait_for(
                    self.transcription_queue.get(), timeout=1.0
                )
                
                # Transcribe audio
                text = await self._transcribe_audio(audio_data)
                
                if text and text.strip() and len(text.strip()) > 2:
                    # Log conversation
                    entry = {
                        "speaker_id": speaker_id,
                        "text": text.strip(),
                        "timestamp": timestamp,
                        "confidence": safe_float(confidence),
                        "duration": len(audio_data) / self.sample_rate
                    }
                    self.conversation_log.append(entry)
                    
                    # Broadcast transcription
                    await self._broadcast_transcription(
                        speaker_id, text.strip(), timestamp, confidence
                    )
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Transcription processing error: {e}")
        
        logger.info("Transcription processing stopped")
    
    async def _transcribe_audio(self, audio_data: np.ndarray) -> str:
        """Transcribe audio using Whisper"""
        try:
            # Save to temp file
            temp_file = await asyncio.to_thread(self._save_temp_audio, audio_data)
            if not temp_file:
                return ""
            
            try:
                # Transcribe with Whisper
                segments, info = await asyncio.to_thread(
                    self.whisper_model.transcribe,
                    temp_file,
                    language="en",
                    beam_size=1,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=300,
                        speech_pad_ms=30
                    )
                )
                
                # Extract valid text
                valid_segments = []
                for segment in segments:
                    if (segment.text and 
                        len(segment.text.strip()) > 2 and
                        segment.avg_logprob > -1.5):  # More lenient threshold
                        
                        cleaned_text = segment.text.strip()
                        # Remove repetitive patterns
                        words = cleaned_text.split()
                        if len(words) > 3:
                            # Check for excessive repetition
                            unique_words = len(set(words))
                            if unique_words / len(words) > 0.3:  # At least 30% unique words
                                valid_segments.append(cleaned_text)
                        else:
                            valid_segments.append(cleaned_text)
                
                result = " ".join(valid_segments)
                return result if len(result.strip()) > 2 else ""
                
            finally:
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
    
    def _save_temp_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """Save audio to temporary WAV file"""
        try:
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            
            with wave.open(temp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                
                # Convert to int16
                audio_int16 = np.clip(audio_data * 32767, -32767, 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())
            
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Error saving temp audio: {e}")
            return None
    
    # Broadcasting methods
    async def _broadcast_speech_start(self, timestamp: float):
        """Broadcast speech detection start"""
        await self._broadcast_message({
            "type": "speech_detected",
            "action": "started", 
            "timestamp": timestamp
        })
    
    async def _broadcast_speech_end(self, timestamp: float):
        """Broadcast speech detection end"""
        await self._broadcast_message({
            "type": "speech_detected",
            "action": "ended",
            "timestamp": timestamp
        })
    
    async def _broadcast_transcription(self, speaker_id: str, text: str, timestamp: float, confidence: float):
        """Broadcast transcription result"""
        message = {
            "type": "transcription",
            "speaker_id": speaker_id,
            "text": text,
            "timestamp": timestamp,
            "confidence": safe_float(confidence)
        }
        await self._broadcast_message(message)
        logger.info(f"{speaker_id}: {text}")
    
    async def _broadcast_speaker_stats(self):
        """Broadcast current speaker statistics"""
        if not self.meeting_start_time:
            return
        
        current_time = time.time()
        meeting_duration = current_time - self.meeting_start_time
        
        speaker_data = []
        total_speaking_time = 0.0
        
        for speaker_id, profile in self.speakers.items():
            talk_time = safe_float(profile.total_time)
            total_speaking_time += talk_time
            
            # Calculate percentage - avoid division by zero
            if meeting_duration > 0:
                percentage = safe_float((talk_time / meeting_duration) * 100.0)
            else:
                percentage = 0.0
            
            speaker_info = {
                "speaker_id": speaker_id,
                "talk_time_seconds": talk_time,
                "percentage": percentage,
                "is_speaking": profile.is_speaking,
                "speech_segments": profile.speech_segments,
                "confidence_score": safe_float(profile.confidence_score)
            }
            speaker_data.append(speaker_info)
        
        # Sort by talk time
        speaker_data.sort(key=lambda x: x["talk_time_seconds"], reverse=True)
        
        message = {
            "type": "speaker_stats",
            "speakers": speaker_data,
            "meeting_duration": safe_float(meeting_duration),
            "total_speaking_time": safe_float(total_speaking_time),
            "silence_time": safe_float(meeting_duration - total_speaking_time),
            "timestamp": current_time
        }
        
        await self._broadcast_message(message)
    
    async def _broadcast_message(self, message: dict):
        """Broadcast message to all connected WebSocket clients"""
        if not self.websocket_connections:
            return
        
        disconnected = []
        json_message = safe_json_dumps(message)
        
        for ws in self.websocket_connections:
            try:
                await ws.send_text(json_message)
            except Exception as e:
                logger.debug(f"Failed to send to websocket: {e}")
                disconnected.append(ws)
        
        # Clean up disconnected clients
        for ws in disconnected:
            if ws in self.websocket_connections:
                self.websocket_connections.remove(ws)
    
    async def _timer_updates(self):
        """Send periodic timer and status updates"""
        while self.is_recording and self.meeting_start_time:
            await asyncio.sleep(2.0)  # Update every 2 seconds
            
            current_time = time.time()
            meeting_duration = current_time - self.meeting_start_time
            
            message = {
                "type": "timer_update",
                "meeting_duration": safe_float(meeting_duration),
                "timestamp": current_time,
                "total_speakers": len(self.speakers),
                "active_speakers": sum(1 for s in self.speakers.values() if s.is_speaking),
                "queue_sizes": {
                    "segments": self.segment_queue.qsize(),
                    "transcriptions": self.transcription_queue.qsize()
                }
            }
            
            await self._broadcast_message(message)
    
    async def _periodic_stats_broadcast(self):
        """Broadcast stats periodically"""
        while self.is_recording:
            await asyncio.sleep(5.0)  # Every 5 seconds
            await self._broadcast_speaker_stats()
    
    # Audio stream management
    async def _start_audio_stream(self):
        """Start audio input stream"""
        try:
            with self._stream_lock:
                if self.audio_stream is not None:
                    return
                
                self._audio_loop = asyncio.get_event_loop()
                
                self.audio_stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    callback=self.audio_callback,
                    blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
                    dtype=np.float32,
                    latency='low'
                )
                
                self.audio_stream.start()
                logger.info(f"Audio stream started: {self.sample_rate}Hz, {self.channels} channel(s)")
                
        except Exception as e:
            logger.error(f"Error starting audio stream: {e}")
            raise
    
    async def _stop_audio_stream(self):
        """Stop audio input stream"""
        with self._stream_lock:
            if self.audio_stream is not None:
                try:
                    self.audio_stream.stop()
                    self.audio_stream.close()
                    logger.info("Audio stream stopped")
                except Exception as e:
                    logger.warning(f"Error stopping audio stream: {e}")
                finally:
                    self.audio_stream = None
                    self._audio_loop = None
    
    # Main lifecycle methods
    async def start_recording(self):
        """Start recording session"""
        if self.is_recording:
            logger.warning("Recording already active")
            return
        
        # Clean stop any existing stream
        await self._stop_audio_stream()
        
        logger.info("Starting enhanced recording session...")
        self.is_recording = True
        self.meeting_start_time = time.time()
        
        # Reset all state
        self.speakers.clear()
        self.next_speaker_index = 1
        self.conversation_log.clear()
        
        # Reset VAD state
        self.in_speech = False
        self.speech_count = 0
        self.silence_count = 0
        self.current_segment = []
        self.ring_pre.clear()
        
        # Clear buffers
        with self._buffer_lock:
            self.audio_buffer.clear()
        
        # Clear queues
        while not self.segment_queue.empty():
            try:
                self.segment_queue.get_nowait()
            except:
                break
        
        while not self.transcription_queue.empty():
            try:
                self.transcription_queue.get_nowait()
            except:
                break
        
        # Start audio stream
        await self._start_audio_stream()
        
        # Start processing tasks
        self.segment_processor_task = asyncio.create_task(self._process_segments())
        self.transcription_task = asyncio.create_task(self._process_transcriptions())
        self.timer_task = asyncio.create_task(self._timer_updates())
        self.stats_task = asyncio.create_task(self._periodic_stats_broadcast())
        
        logger.info("Enhanced recording session started successfully")
    
    async def stop_recording(self):
        """Stop recording session"""
        logger.info("Stopping recording session...")
        self.is_recording = False
        
        # Stop audio stream first
        await self._stop_audio_stream()
        
        # Stop all tasks
        tasks = [
            (self.segment_processor_task, "segment processor"),
            (self.transcription_task, "transcription processor"),
            (self.timer_task, "timer updates"),
            (self.stats_task, "stats broadcaster")
        ]
        
        for task, name in tasks:
            if task and not task.done():
                try:
                    await asyncio.wait_for(task, timeout=3.0)
                    logger.info(f"Stopped {name}")
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout stopping {name}, cancelling...")
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                except Exception as e:
                    logger.error(f"Error stopping {name}: {e}")
        
        # Final stats broadcast
        await self._broadcast_speaker_stats()
        
        # Broadcast final summary
        await self._broadcast_final_summary()
        
        logger.info("Recording session stopped")
    
    async def _broadcast_final_summary(self):
        """Broadcast final meeting summary"""
        if not self.meeting_start_time:
            return
        
        total_duration = time.time() - self.meeting_start_time
        
        summary = {
            "type": "meeting_summary",
            "total_duration": safe_float(total_duration),
            "total_speakers": len(self.speakers),
            "total_segments": sum(s.speech_segments for s in self.speakers.values()),
            "total_transcriptions": len(self.conversation_log),
            "speaker_breakdown": [s.get_stats() for s in self.speakers.values()],
            "timestamp": time.time()
        }
        
        await self._broadcast_message(summary)
    
    async def cleanup(self):
        """Cleanup all resources"""
        logger.info("Starting cleanup...")
        await self.stop_recording()
        
        # Clear all data
        self.speakers.clear()
        self.conversation_log.clear()
        
        with self._buffer_lock:
            self.audio_buffer.clear()
        
        self.current_segment.clear()
        self.ring_pre.clear()
        
        logger.info("Cleanup completed")
    
    def get_conversation_insights(self) -> dict:
        """Generate conversation insights and statistics"""
        if not self.meeting_start_time:
            return {"error": "No active meeting"}
        
        current_time = time.time()
        meeting_duration = current_time - self.meeting_start_time
        
        insights = {
            "meeting_duration": safe_float(meeting_duration),
            "total_speakers": len(self.speakers),
            "total_segments": len(self.conversation_log),
            "speakers": []
        }
        
        # Speaker statistics
        total_speaking_time = 0.0
        for speaker_id, profile in self.speakers.items():
            speaking_time = safe_float(profile.total_time)
            total_speaking_time += speaking_time
            
            # Get transcriptions for this speaker
            speaker_transcriptions = [
                entry for entry in self.conversation_log 
                if entry["speaker_id"] == speaker_id
            ]
            
            # Calculate word count
            word_count = sum(
                len(entry["text"].split()) 
                for entry in speaker_transcriptions
            )
            
            speaker_stats = {
                "speaker_id": speaker_id,
                "speaking_time": speaking_time,
                "percentage": safe_float((speaking_time / max(meeting_duration, 1)) * 100),
                "speech_segments": profile.speech_segments,
                "transcriptions": len(speaker_transcriptions),
                "word_count": word_count,
                "confidence_score": safe_float(profile.confidence_score),
                "is_speaking": profile.is_speaking,
                "recent_transcriptions": speaker_transcriptions[-3:]  # Last 3 transcriptions
            }
            insights["speakers"].append(speaker_stats)
        
        # Sort by speaking time
        insights["speakers"].sort(key=lambda x: x["speaking_time"], reverse=True)
        
        # Overall stats
        insights["total_speaking_time"] = safe_float(total_speaking_time)
        insights["silence_time"] = safe_float(meeting_duration - total_speaking_time)
        insights["silence_percentage"] = safe_float(
            ((meeting_duration - total_speaking_time) / max(meeting_duration, 1)) * 100
        )
        
        return insights


# Global assistant instance
meeting_assistant = EnhancedMeetingAssistant()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown"""
    await meeting_assistant.cleanup()

# -------------------------
# API Endpoints
# -------------------------

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Enhanced Real-Time Meeting Assistant API is running",
        "version": "3.0.0",
        "features": [
            "enhanced_speaker_detection",
            "multiple_vad_methods",
            "robust_feature_extraction",
            "nan_safe_calculations",
            "conversation_insights",
            "improved_transcription_quality"
        ],
        "models": {
            "whisper": "base.en" if meeting_assistant.whisper_model else "not_loaded",
            "vad_levels": len(meeting_assistant.vad.vad_levels)
        }
    }

@app.post("/api/meeting/start")
async def start_meeting():
    """Start a new meeting recording session"""
    try:
        await meeting_assistant.start_recording()
        return {
            "status": "success",
            "message": "Enhanced recording started with multi-speaker detection",
            "timestamp": time.time(),
            "session_id": f"session_{int(time.time())}"
        }
    except Exception as e:
        logger.error(f"Error starting meeting: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/meeting/stop")
async def stop_meeting():
    """Stop the current meeting recording session"""
    try:
        await meeting_assistant.stop_recording()
        return {
            "status": "success",
            "message": "Recording stopped successfully",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error stopping meeting: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/meeting/status")
async def get_meeting_status():
    """Get detailed meeting status and statistics"""
    current_time = time.time()
    meeting_duration = (
        current_time - meeting_assistant.meeting_start_time 
        if meeting_assistant.meeting_start_time else 0
    )
    
    # VAD debug info
    vad_info = {
        "in_speech": meeting_assistant.in_speech,
        "speech_count": meeting_assistant.speech_count,
        "silence_count": meeting_assistant.silence_count,
        "current_segment_length": len(meeting_assistant.current_segment),
        "similarity_threshold": meeting_assistant.similarity_threshold,
        "vad_methods": len(meeting_assistant.vad.vad_levels)
    }
    
    # Speaker details
    speakers_detail = {}
    for speaker_id, profile in meeting_assistant.speakers.items():
        speakers_detail[speaker_id] = {
            "total_time": safe_float(profile.total_time),
            "speech_segments": profile.speech_segments,
            "is_speaking": profile.is_speaking,
            "confidence_score": safe_float(profile.confidence_score),
            "last_seen": profile.last_seen,
            "embedding_count": len(profile.recent_embeddings)
        }
    
    status = {
        "is_recording": meeting_assistant.is_recording,
        "meeting_duration": safe_float(meeting_duration),
        "connected_clients": len(meeting_assistant.websocket_connections),
        "total_speakers": len(meeting_assistant.speakers),
        "total_transcriptions": len(meeting_assistant.conversation_log),
        "timestamp": current_time,
        "vad_state": vad_info,
        "speakers": speakers_detail,
        "queue_sizes": {
            "segment_queue": meeting_assistant.segment_queue.qsize(),
            "transcription_queue": meeting_assistant.transcription_queue.qsize()
        }
    }
    
    return status

@app.get("/api/meeting/conversation-insights")
async def get_conversation_insights():
    """Get detailed conversation insights and analytics"""
    try:
        insights = meeting_assistant.get_conversation_insights()
        return {
            "status": "success",
            "data": insights,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error getting conversation insights: {e}")
        return {
            "status": "error", 
            "message": str(e),
            "data": {
                "meeting_duration": 0,
                "total_speakers": 0,
                "speakers": []
            }
        }

@app.get("/api/debug/audio")
async def debug_audio_status():
    """Debug endpoint for audio system diagnostics"""
    try:
        # Audio buffer info
        buffer_size = 0
        with meeting_assistant._buffer_lock:
            buffer_size = len(meeting_assistant.audio_buffer)
        
        # VAD configuration
        vad_config = {
            "energy_threshold_base": meeting_assistant.vad.energy_threshold_base,
            "energy_multiplier": meeting_assistant.vad.energy_multiplier,
            "noise_floor": safe_float(meeting_assistant.vad.noise_floor),
            "trigger_frames": meeting_assistant.vad.trigger_frames,
            "release_frames": meeting_assistant.vad.release_frames,
            "energy_history_length": len(meeting_assistant.vad.energy_history),
            "vad_methods_available": len(meeting_assistant.vad.vad_levels)
        }
        
        # Audio stream info
        stream_info = {
            "stream_active": meeting_assistant.audio_stream is not None,
            "sample_rate": meeting_assistant.sample_rate,
            "channels": meeting_assistant.channels,
            "frame_size": meeting_assistant.vad.frame_size,
            "process_frame_size": meeting_assistant.process_frame_size
        }
        
        # Processing state
        processing_state = {
            "in_speech": meeting_assistant.in_speech,
            "current_segment_samples": len(meeting_assistant.current_segment),
            "pre_speech_buffer_samples": len(meeting_assistant.ring_pre),
            "speech_count": meeting_assistant.speech_count,
            "silence_count": meeting_assistant.silence_count
        }
        
        return {
            "audio_buffer_size": buffer_size,
            "vad_config": vad_config,
            "stream_info": stream_info,
            "processing_state": processing_state,
            "queue_sizes": {
                "segment_queue": meeting_assistant.segment_queue.qsize(),
                "transcription_queue": meeting_assistant.transcription_queue.qsize()
            },
            "speaker_stats": {
                "total_speakers": len(meeting_assistant.speakers),
                "similarity_threshold": meeting_assistant.similarity_threshold,
                "max_speakers": meeting_assistant.max_speakers
            }
        }
        
    except Exception as e:
        return {"error": str(e), "timestamp": time.time()}

@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    meeting_assistant.websocket_connections.append(websocket)
    logger.info(f"WebSocket connected. Total clients: {len(meeting_assistant.websocket_connections)}")
    
    # Send connection confirmation
    await websocket.send_text(safe_json_dumps({
        "type": "connection_established",
        "message": "Connected to Enhanced Meeting Assistant",
        "version": "3.0.0",
        "features": ["multi_speaker", "enhanced_vad", "conversation_insights"],
        "timestamp": time.time()
    }))
    
    try:
        while True:
            # Listen for client messages
            message = await websocket.receive_text()
            try:
                data = json.loads(message)
                logger.debug(f"Received client message: {data}")
                
                # Handle client requests
                if data.get("type") == "request_status":
                    status = await get_meeting_status()
                    await websocket.send_text(safe_json_dumps({
                        "type": "status_response",
                        "data": status
                    }))
                    
            except json.JSONDecodeError:
                logger.debug(f"Received non-JSON message: {message}")
                
    except WebSocketDisconnect:
        if websocket in meeting_assistant.websocket_connections:
            meeting_assistant.websocket_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Remaining: {len(meeting_assistant.websocket_connections)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0", 
        port=8001,
        log_level="info",
        access_log=True
    )