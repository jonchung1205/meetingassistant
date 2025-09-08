import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import './App.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [speakerData, setSpeakerData] = useState([]);
  const [meetingDuration, setMeetingDuration] = useState(0);
  const [totalSpeakingTime, setTotalSpeakingTime] = useState(0);
  const [silenceTime, setSilenceTime] = useState(0);
  const [prompts, setPrompts] = useState([]);
  const [agenda, setAgenda] = useState('');
  const [agendaItems, setAgendaItems] = useState([]);
  
  // Enhanced real-time states
  const [currentSpeaker, setCurrentSpeaker] = useState(null);
  const [activeSpeakers, setActiveSpeakers] = useState(new Set()); // Track multiple active speakers
  const [transcriptions, setTranscriptions] = useState([]);
  const [lastActivity, setLastActivity] = useState(null);
  const [speakerActivity, setSpeakerActivity] = useState({});
  const [realtimeStats, setRealtimeStats] = useState({
    meeting_duration: 0,
    total_speaking_time: 0,
    silence_time: 0
  });
  
  // Enhanced conversation states
  const [conversationMomentum, setConversationMomentum] = useState(0);
  const [silenceState, setSilenceState] = useState({ active: false, category: null, duration: 0 });
  const [speakerTransitions, setSpeakerTransitions] = useState(0);
  const [dominantSpeaker, setDominantSpeaker] = useState(null);
  const [conversationInsights, setConversationInsights] = useState({});
  const [recentSpeakerSequence, setRecentSpeakerSequence] = useState([]);
  
  // New multi-speaker overlap detection
  const [speakerOverlaps, setSpeakerOverlaps] = useState([]);
  const [conversationFlow, setConversationFlow] = useState({ transitions: {}, patterns: [] });
  const [speakerProfiles, setSpeakerProfiles] = useState({});
  
  // Connection and performance states
  const [connectionQuality, setConnectionQuality] = useState('good');
  const [lastMessageTime, setLastMessageTime] = useState(Date.now());
  const [messageCount, setMessageCount] = useState(0);
  const [backendVersion, setBackendVersion] = useState('2.0.0');
  const [serverFeatures, setServerFeatures] = useState([]);
  
  const ws = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const isConnectingRef = useRef(false);
  const transcriptionRef = useRef(null);
  const performanceTimerRef = useRef(null);
  const insightsTimerRef = useRef(null);
  const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

  // Enhanced performance monitoring
  useEffect(() => {
    performanceTimerRef.current = setInterval(() => {
      const now = Date.now();
      const timeSinceLastMessage = now - lastMessageTime;
      
      if (isRecording && timeSinceLastMessage > 5000) {
        setConnectionQuality('poor');
      } else if (isRecording && timeSinceLastMessage > 2000) {
        setConnectionQuality('fair');
      } else if (isRecording) {
        setConnectionQuality('good');
      }
    }, 1000);

    return () => {
      if (performanceTimerRef.current) {
        clearInterval(performanceTimerRef.current);
      }
    };
  }, [lastMessageTime, isRecording]);

  // Enhanced conversation insights fetching
  useEffect(() => {
    if (isRecording) {
      insightsTimerRef.current = setInterval(async () => {
        try {
          const response = await fetch(`${backendUrl}/api/meeting/conversation-insights`);
          if (response.ok) {
            const insights = await response.json();
            setConversationInsights(insights);
            
            if (insights.recent_speaker_sequence) {
              setRecentSpeakerSequence(insights.recent_speaker_sequence);
            }
            
            if (insights.speaker_transitions) {
              setConversationFlow({
                transitions: insights.speaker_transitions,
                patterns: insights.conversation_patterns || []
              });
            }
            
            // Update speaker dominance visualization
            if (insights.speaker_dominance) {
              setSpeakerProfiles(prev => {
                const updated = { ...prev };
                Object.entries(insights.speaker_dominance).forEach(([speakerId, dominance]) => {
                  if (!updated[speakerId]) {
                    updated[speakerId] = {};
                  }
                  updated[speakerId].dominance = dominance;
                });
                return updated;
              });
            }
          }
        } catch (error) {
          console.warn('Failed to fetch conversation insights:', error);
        }
      }, 2000); // Every 2 seconds for more responsive updates
    }

    return () => {
      if (insightsTimerRef.current) {
        clearInterval(insightsTimerRef.current);
      }
    };
  }, [isRecording, backendUrl]);

  useEffect(() => {
    connectToWebSocket();
    
    return () => {
      isConnectingRef.current = false;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      if (performanceTimerRef.current) {
        clearInterval(performanceTimerRef.current);
      }
      if (insightsTimerRef.current) {
        clearInterval(insightsTimerRef.current);
      }
      if (ws.current) {
        ws.current.onopen = null;
        ws.current.onmessage = null;
        ws.current.onclose = null;
        ws.current.onerror = null;
        
        if (ws.current.readyState === WebSocket.OPEN || ws.current.readyState === WebSocket.CONNECTING) {
          ws.current.close();
        }
        ws.current = null;
      }
    };
  }, []);

  // Auto-scroll transcriptions
  useEffect(() => {
    if (transcriptionRef.current) {
      transcriptionRef.current.scrollTop = transcriptionRef.current.scrollHeight;
    }
  }, [transcriptions]);

  const connectToWebSocket = useCallback(() => {
    if (isConnectingRef.current || (ws.current && ws.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    if (ws.current) {
      ws.current.onopen = null;
      ws.current.onmessage = null;
      ws.current.onclose = null;
      ws.current.onerror = null;
      if (ws.current.readyState === WebSocket.OPEN) {
        ws.current.close();
      }
      ws.current = null;
    }

    const wsUrl = backendUrl.replace('http', 'ws') + '/api/ws';
    isConnectingRef.current = true;
    
    try {
      console.log('Connecting to Enhanced WebSocket:', wsUrl);
      ws.current = new WebSocket(wsUrl);
      
      ws.current.onopen = () => {
        console.log('Connected to Enhanced WebSocket');
        isConnectingRef.current = false;
        setIsConnected(true);
        setConnectionQuality('good');
        
        if (ws.current && ws.current.readyState === WebSocket.OPEN) {
          ws.current.send(JSON.stringify({ type: 'ping' }));
        }
      };
      
      ws.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setLastMessageTime(Date.now());
          setMessageCount(prev => prev + 1);
          handleWebSocketMessage(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
      
      ws.current.onclose = (event) => {
        console.log('Enhanced WebSocket connection closed', event.code, event.reason);
        isConnectingRef.current = false;
        setIsConnected(false);
        setConnectionQuality('poor');
        
        if (event.code !== 1000 && event.code !== 1001) {
          if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
          }
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connectToWebSocket();
          }, 3000);
        }
      };
      
      ws.current.onerror = (error) => {
        console.error('Enhanced WebSocket error:', error);
        isConnectingRef.current = false;
        setIsConnected(false);
        setConnectionQuality('poor');
      };
      
    } catch (error) {
      console.error('Failed to create Enhanced WebSocket connection:', error);
      isConnectingRef.current = false;
      setIsConnected(false);
      setConnectionQuality('poor');
      
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      reconnectTimeoutRef.current = setTimeout(() => {
        connectToWebSocket();
      }, 3000);
    }
  }, [backendUrl]);

  const handleWebSocketMessage = useCallback((data) => {
    switch (data.type) {
      case 'connection_established':
        if (data.version) {
          setBackendVersion(data.version);
        }
        if (data.features) {
          setServerFeatures(data.features);
        }
        console.log('Enhanced connection established:', data.message);
        break;
        
      case 'speaker_activity':
        handleEnhancedSpeakerActivity(data);
        break;
        
      case 'speaker_stats':
        handleEnhancedSpeakerStats(data);
        break;
        
      case 'transcription':
        handleEnhancedTranscription(data);
        break;
        
      case 'timer_update':
        handleEnhancedTimerUpdate(data);
        break;
        
      case 'silence':
        handleEnhancedSilenceEvent(data);
        break;
        
      case 'speaker_overlap':
        handleSpeakerOverlap(data);
        break;
        
      case 'prompt':
        setPrompts(prev => [...prev, data]);
        break;
        
      default:
        console.log('Unknown message type:', data.type, data);
    }
  }, []);

  const handleEnhancedSpeakerActivity = useCallback((data) => {
    const { speaker_id, action, timestamp, conversation_momentum, overlap_detected } = data;
    
    setLastActivity({ speaker_id, action, timestamp });
    
    // Update conversation momentum
    if (conversation_momentum !== undefined) {
      setConversationMomentum(conversation_momentum);
    }
    
    // Handle multi-speaker scenarios
    setActiveSpeakers(prev => {
      const updated = new Set(prev);
      if (action === 'started') {
        updated.add(speaker_id);
        setCurrentSpeaker(speaker_id); // Primary speaker
      } else if (action === 'stopped') {
        updated.delete(speaker_id);
        if (speaker_id === currentSpeaker) {
          // Set new primary speaker if available
          const remaining = Array.from(updated);
          setCurrentSpeaker(remaining.length > 0 ? remaining[0] : null);
        }
      }
      return updated;
    });
    
    // Track speaker overlaps
    if (overlap_detected) {
      setSpeakerOverlaps(prev => {
        const newOverlap = {
          id: `overlap-${timestamp}`,
          speakers: data.overlapping_speakers || [speaker_id],
          timestamp,
          duration: data.overlap_duration || 0
        };
        return [newOverlap, ...prev.slice(0, 19)]; // Keep last 20 overlaps
      });
    }
    
    // Enhanced speaker activity tracking
    setSpeakerActivity(prev => ({
      ...prev,
      [speaker_id]: {
        ...prev[speaker_id],
        last_action: action,
        last_timestamp: timestamp,
        is_active: action === 'started',
        confidence: data.confidence || 1.0,
        audio_quality: data.audio_quality || 'good'
      }
    }));
  }, [currentSpeaker]);

  const handleEnhancedSpeakerStats = useCallback((data) => {
    const { 
      speakers, 
      meeting_duration, 
      total_speaking_time, 
      silence_time,
      conversation_momentum,
      total_speaker_transitions,
      dominant_speaker
    } = data;
    
    // Enhanced speaker data processing
    const formattedSpeakers = speakers.map(speaker => ({
      speaker_id: speaker.speaker_id,
      talk_time_in_seconds: speaker.talk_time_seconds,
      percentage: speaker.percentage,
      is_speaking: speaker.is_speaking,
      speech_segments: speaker.speech_segments || 0,
      confidence_score: speaker.confidence_score || 1.0,
      avg_speaking_duration: speaker.avg_speaking_duration || 0,
      avg_pitch: speaker.avg_pitch,
      avg_energy: speaker.avg_energy,
      speaking_pattern_score: speaker.speaking_pattern_score || 0.5,
      // New enhanced metrics
      voice_consistency: speaker.voice_consistency || 0.8,
      interrupt_count: speaker.interrupt_count || 0,
      interrupted_count: speaker.interrupted_count || 0
    }));
    
    setSpeakerData(formattedSpeakers);
    setMeetingDuration(meeting_duration);
    setTotalSpeakingTime(total_speaking_time);
    setSilenceTime(silence_time);
    
    // Enhanced metrics
    if (conversation_momentum !== undefined) {
      setConversationMomentum(conversation_momentum);
    }
    if (total_speaker_transitions !== undefined) {
      setSpeakerTransitions(total_speaker_transitions);
    }
    if (dominant_speaker) {
      setDominantSpeaker(dominant_speaker);
    }
    
    // Update speaker profiles
    setSpeakerProfiles(prev => {
      const updated = { ...prev };
      formattedSpeakers.forEach(speaker => {
        updated[speaker.speaker_id] = {
          ...updated[speaker.speaker_id],
          ...speaker,
          last_updated: Date.now()
        };
      });
      return updated;
    });
    
    setRealtimeStats({
      meeting_duration,
      total_speaking_time,
      silence_time
    });
  }, []);

  const handleEnhancedTranscription = useCallback((data) => {
    const { 
      speaker_id, 
      text, 
      timestamp, 
      confidence,
      word_count,
      conversation_momentum,
      recent_speaker_sequence,
      speaking_rate,
      sentiment_score
    } = data;
    
    setTranscriptions(prev => {
      const newTranscription = {
        id: `${timestamp}-${Math.random()}`,
        speaker_id,
        text,
        timestamp,
        time: new Date(timestamp * 1000).toLocaleTimeString(),
        confidence: confidence || 1.0,
        word_count: word_count || 0,
        conversation_momentum: conversation_momentum || 0,
        // Enhanced transcription features
        speaking_rate: speaking_rate || null,
        sentiment_score: sentiment_score || null,
        is_question: text.trim().endsWith('?'),
        text_length: text.length
      };
      
      // Keep only last 150 transcriptions for better performance
      const updated = [...prev, newTranscription];
      return updated.slice(-150);
    });

    // Update conversation momentum
    if (conversation_momentum !== undefined) {
      setConversationMomentum(conversation_momentum);
    }
    
    // Update recent speaker sequence
    if (recent_speaker_sequence) {
      setRecentSpeakerSequence(recent_speaker_sequence);
    }
  }, []);

  const handleEnhancedTimerUpdate = useCallback((data) => {
    const { 
      meeting_duration, 
      active_speakers, 
      total_speakers,
      conversation_momentum,
      silence_ratio,
      speaking_efficiency
    } = data;
    
    setMeetingDuration(meeting_duration);
    
    if (conversation_momentum !== undefined) {
      setConversationMomentum(conversation_momentum);
    }
  }, []);

  const handleEnhancedSilenceEvent = useCallback((data) => {
    const { state, timestamp, duration, silence_category, conversation_momentum } = data;
    
    setSilenceState({
      active: state === 'started',
      category: silence_category || null,
      duration: duration || 0,
      timestamp: timestamp
    });

    if (conversation_momentum !== undefined) {
      setConversationMomentum(conversation_momentum);
    }
  }, []);

  const handleSpeakerOverlap = useCallback((data) => {
    const { speakers, timestamp, duration, overlap_type } = data;
    
    setSpeakerOverlaps(prev => {
      const newOverlap = {
        id: `overlap-${timestamp}-${Math.random()}`,
        speakers: speakers || [],
        timestamp,
        duration: duration || 0,
        type: overlap_type || 'simultaneous',
        time: new Date(timestamp * 1000).toLocaleTimeString()
      };
      return [newOverlap, ...prev.slice(0, 24)]; // Keep last 25 overlaps
    });
  }, []);

  const startMeeting = async () => {
    try {
      const response = await fetch(`${backendUrl}/api/meeting/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (response.ok) {
        setIsRecording(true);
        // Reset all states
        setPrompts([]);
        setSpeakerData([]);
        setMeetingDuration(0);
        setTotalSpeakingTime(0);
        setSilenceTime(0);
        setTranscriptions([]);
        setCurrentSpeaker(null);
        setActiveSpeakers(new Set());
        setLastActivity(null);
        setSpeakerActivity({});
        setRealtimeStats({ meeting_duration: 0, total_speaking_time: 0, silence_time: 0 });
        setMessageCount(0);
        
        // Reset enhanced states
        setConversationMomentum(0);
        setSilenceState({ active: false, category: null, duration: 0 });
        setSpeakerTransitions(0);
        setDominantSpeaker(null);
        setConversationInsights({});
        setRecentSpeakerSequence([]);
        setSpeakerOverlaps([]);
        setConversationFlow({ transitions: {}, patterns: [] });
        setSpeakerProfiles({});
        
        console.log('Enhanced meeting started successfully');
      } else {
        const errorData = await response.json().catch(() => ({ message: 'Unknown error' }));
        console.error('Failed to start enhanced meeting:', errorData);
        alert(`Failed to start meeting: ${errorData.message || 'Please check if the backend is running.'}`);
      }
    } catch (error) {
      console.error('Error starting enhanced meeting:', error);
      alert('Error starting meeting. Please check your connection and ensure the enhanced backend server is running.');
    }
  };

  const stopMeeting = async () => {
    try {
      const response = await fetch(`${backendUrl}/api/meeting/stop`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (response.ok) {
        setIsRecording(false);
        setCurrentSpeaker(null);
        setActiveSpeakers(new Set());
        setSpeakerActivity({});
        setSilenceState({ active: false, category: null, duration: 0 });
        console.log('Enhanced meeting stopped successfully');
      } else {
        console.error('Failed to stop enhanced meeting');
      }
    } catch (error) {
      console.error('Error stopping enhanced meeting:', error);
    }
  };

  const setMeetingAgenda = async () => {
    if (!agenda.trim()) return;
    
    const items = agenda.split('\n').filter(item => item.trim()).map(item => item.trim());
    
    try {
      const response = await fetch(`${backendUrl}/api/meeting/agenda`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ items }),
      });
      
      if (response.ok) {
        setAgendaItems(items);
        console.log('Agenda set successfully');
      } else {
        console.error('Failed to set agenda');
      }
    } catch (error) {
      console.error('Error setting agenda:', error);
    }
  };

  const dismissPrompt = (index) => {
    setPrompts(prev => prev.filter((_, i) => i !== index));
  };

  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  const formatTimeDetailed = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  };

  const getConnectionStatusIcon = () => {
    switch (connectionQuality) {
      case 'good': return '🟢';
      case 'fair': return '🟡';
      case 'poor': return '🔴';
      default: return '⚫';
    }
  };

  const getConnectionStatusText = () => {
    const base = isConnected ? 'Connected' : 'Disconnected';
    if (isConnected && isRecording) {
      return `${base} (${connectionQuality})`;
    }
    return base;
  };

  const getMomentumIndicator = () => {
    if (conversationMomentum > 0.7) return { text: 'High Energy', icon: '⚡', color: '#10B981' };
    if (conversationMomentum > 0.4) return { text: 'Active', icon: '🎯', color: '#3B82F6' };
    if (conversationMomentum > 0.2) return { text: 'Moderate', icon: '💭', color: '#F59E0B' };
    return { text: 'Quiet', icon: '😴', color: '#6B7280' };
  };

  const getSilenceDisplay = () => {
    if (!silenceState.active) return null;
    
    const categories = {
      brief_pause: { text: 'Brief Pause', color: '#10B981' },
      natural_pause: { text: 'Natural Break', color: '#3B82F6' },
      long_pause: { text: 'Long Pause', color: '#F59E0B' },
      extended_silence: { text: 'Extended Silence', color: '#EF4444' }
    };
    
    return categories[silenceState.category] || { text: 'Silence', color: '#6B7280' };
  };

  const getSpeakerStatusColor = (speakerId) => {
    if (activeSpeakers.has(speakerId)) {
      return speakerId === currentSpeaker ? '#10B981' : '#3B82F6'; // Primary vs secondary
    }
    return '#6B7280';
  };

  // Enhanced chart data with multi-speaker support
  const chartData = {
    labels: speakerData.map(speaker => speaker.speaker_id),
    datasets: [
      {
        label: 'Speaking Time (seconds)',
        data: speakerData.map(speaker => speaker.talk_time_in_seconds),
        backgroundColor: speakerData.map((speaker, index) => {
          const colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899', '#06B6D4'];
          const baseColor = colors[index % colors.length];
          
          // Multi-level visual indicators
          let opacity = Math.max(0.5, (speaker.confidence_score || 1.0));
          if (activeSpeakers.has(speaker.speaker_id)) {
            opacity = 1.0; // Full opacity for active speakers
          }
          if (speaker.speaker_id === currentSpeaker) {
            return baseColor; // Full color for primary speaker
          }
          
          return baseColor + Math.round(opacity * 255).toString(16).padStart(2, '0');
        }),
        borderColor: speakerData.map((speaker, index) => {
          const colors = ['#2563EB', '#059669', '#D97706', '#DC2626', '#7C3AED', '#DB2777', '#0891B2'];
          return colors[index % colors.length];
        }),
        borderWidth: speakerData.map(speaker => {
          if (speaker.speaker_id === currentSpeaker) return 4;
          if (activeSpeakers.has(speaker.speaker_id)) return 3;
          return 1;
        }),
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Enhanced Multi-Speaker Real-Time Analysis',
        font: {
          size: 18,
        },
      },
      tooltip: {
        callbacks: {
          afterLabel: function(context) {
            const speaker = speakerData[context.dataIndex];
            const lines = [];
            
            if (speaker.confidence_score) {
              lines.push(`Voice ID Confidence: ${Math.round(speaker.confidence_score * 100)}%`);
            }
            if (speaker.voice_consistency) {
              lines.push(`Voice Consistency: ${Math.round(speaker.voice_consistency * 100)}%`);
            }
            if (speaker.avg_speaking_duration) {
              lines.push(`Avg Duration: ${speaker.avg_speaking_duration.toFixed(1)}s`);
            }
            if (speaker.speech_segments) {
              lines.push(`Speech Segments: ${speaker.speech_segments}`);
            }
            if (speaker.interrupt_count > 0) {
              lines.push(`Interruptions Made: ${speaker.interrupt_count}`);
            }
            if (speaker.interrupted_count > 0) {
              lines.push(`Times Interrupted: ${speaker.interrupted_count}`);
            }
            return lines;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Time (seconds)',
        },
      },
      x: {
        title: {
          display: true,
          text: 'Speakers',
        },
      },
    },
    animation: {
      duration: 200, // Faster animations for real-time feel
    },
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>🎤 Enhanced Multi-Speaker Meeting Assistant v{backendVersion}</h1>
        <div className="header-info">
          <div className="connection-status">
            <span className={`status-indicator ${isConnected ? 'connected' : 'disconnected'} ${connectionQuality}`}>
              {getConnectionStatusIcon()} {getConnectionStatusText()}
            </span>
          </div>
          {isRecording && (
            <div className="performance-stats">
              <span className="message-count">Messages: {messageCount}</span>
              <span className="conversation-momentum" style={{ color: getMomentumIndicator().color }}>
                {getMomentumIndicator().icon} {getMomentumIndicator().text}
              </span>
              {activeSpeakers.size > 1 && (
                <span className="multi-speaker-indicator">
                  👥 {activeSpeakers.size} Active Speakers
                </span>
              )}
            </div>
          )}
        </div>
      </header>

      <main className="main-content">
        {/* Enhanced Control Panel */}
        <div className="control-panel">
          <div className="meeting-controls">
            <button
              onClick={isRecording ? stopMeeting : startMeeting}
              className={`control-button ${isRecording ? 'stop' : 'start'}`}
              disabled={!isConnected}
            >
              {isRecording ? '⏹️ Stop Meeting' : '▶️ Start Meeting'}
            </button>
            
            <div className="meeting-stats">
              <div className="stat">
                <span className="stat-label">Duration:</span>
                <span className="stat-value">{formatTimeDetailed(meetingDuration)}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Speaking:</span>
                <span className="stat-value">{formatTimeDetailed(totalSpeakingTime)}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Silence:</span>
                <span className="stat-value">{formatTimeDetailed(silenceTime)}</span>
              </div>
              {speakerTransitions > 0 && (
                <div className="stat">
                  <span className="stat-label">Transitions:</span>
                  <span className="stat-value">{speakerTransitions}</span>
                </div>
              )}
            </div>
          </div>

          {/* Enhanced Multi-Speaker Status Panel */}
          {isRecording && (
            <div className="current-speaker-status">
              <h3>🎙️ Enhanced Multi-Speaker Live Status</h3>
              
              {/* Active Speakers Display */}
              <div className="active-speakers-panel">
                {activeSpeakers.size > 0 ? (
                  <div className="multi-speaker-display">
                    <div className="speakers-grid">
                      {Array.from(activeSpeakers).map(speakerId => {
                        const isPrimary = speakerId === currentSpeaker;
                        const profile = speakerProfiles[speakerId] || {};
                        const activity = speakerActivity[speakerId] || {};
                        
                        return (
                          <div 
                            key={speakerId} 
                            className={`speaker-card ${isPrimary ? 'primary' : 'secondary'}`}
                            style={{ borderColor: getSpeakerStatusColor(speakerId) }}
                          >
                            <div className="speaker-header">
                              <span className="speaker-name">{speakerId}</span>
                              {isPrimary && <span className="primary-badge">Primary</span>}
                              <div className="speaker-visual">
                                <div className="audio-bars">
                                  <div className="bar"></div>
                                  <div className="bar"></div>
                                  <div className="bar"></div>
                                </div>
                              </div>
                            </div>
                            <div className="speaker-metrics">
                              {profile.confidence_score && (
                                <span className="confidence-metric">
                                  ID: {Math.round(profile.confidence_score * 100)}%
                                </span>
                              )}
                              {profile.voice_consistency && (
                                <span className="consistency-metric">
                                  Voice: {Math.round(profile.voice_consistency * 100)}%
                                </span>
                              )}
                              {activity.audio_quality && (
                                <span className={`quality-metric ${activity.audio_quality}`}>
                                  {activity.audio_quality}
                                </span>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                    
                    {/* Overlap Detection Alert */}
                    {speakerOverlaps.length > 0 && speakerOverlaps[0].timestamp > Date.now() / 1000 - 5 && (
                      <div className="overlap-alert">
                        <span className="overlap-icon">⚠️</span>
                        <span className="overlap-text">
                          Speaker overlap detected: {speakerOverlaps[0].speakers.join(' & ')}
                        </span>
                        <span className="overlap-duration">
                          ({speakerOverlaps[0].duration.toFixed(1)}s)
                        </span>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="listening-state">
                    <div className="listening-icon">👂</div>
                    <span className="inactive-text">
                      {silenceState.active ? (
                        <span style={{ color: getSilenceDisplay()?.color }}>
                          {getSilenceDisplay()?.text} {silenceState.duration > 0 && `(${silenceState.duration.toFixed(1)}s)`}
                        </span>
                      ) : (
                        'Enhanced multi-speaker detection active...'
                      )}
                    </span>
                  </div>
                )}
              </div>
              
              {/* Enhanced Conversation Flow */}
              {recentSpeakerSequence.length > 0 && (
                <div className="conversation-flow">
                  <h4>Recent Speaker Flow</h4>
                  <div className="speaker-sequence">
                    {recentSpeakerSequence.map((speaker, index) => (
                      <span key={index} className={`sequence-item ${activeSpeakers.has(speaker) ? 'active' : ''} ${speaker === currentSpeaker ? 'current' : ''}`}>
                        {speaker}
                        {index < recentSpeakerSequence.length - 1 && <span className="arrow">→</span>}
                      </span>
                    ))}
                  </div>
                  {conversationFlow.transitions && Object.keys(conversationFlow.transitions).length > 0 && (
                    <div className="transition-patterns">
                      <span className="pattern-label">Common patterns:</span>
                      {Object.entries(conversationFlow.transitions).slice(0, 3).map(([from, transitions], idx) => {
                        const mostCommon = Object.entries(transitions).sort((a, b) => b[1] - a[1])[0];
                        if (mostCommon) {
                          return (
                            <span key={idx} className="pattern-item">
                              {from} → {mostCommon[0]} ({mostCommon[1]}x)
                            </span>
                          );
                        }
                        return null;
                      })}
                    </div>
                  )}
                </div>
              )}
              
              {/* Speaker Activity History with Enhanced Info */}
              {Object.keys(speakerActivity).length > 0 && (
                <div className="speaker-activity-history">
                  <h4>Recent Activity & Quality</h4>
                  <div className="activity-list">
                    {Object.entries(speakerActivity)
                      .sort(([,a], [,b]) => (b.last_timestamp || 0) - (a.last_timestamp || 0))
                      .slice(0, 4)
                      .map(([speakerId, activity]) => {
                        const speakerInfo = speakerData.find(s => s.speaker_id === speakerId);
                        const profile = speakerProfiles[speakerId] || {};
                        return (
                          <div key={speakerId} className={`activity-item ${activity.is_active ? 'active' : 'inactive'}`}>
                            <div className="activity-header">
                              <span className="speaker-name">{speakerId}</span>
                              <span className={`activity-status ${activity.is_active ? 'speaking' : 'silent'}`}>
                                {activity.is_active ? 'Speaking' : 'Silent'}
                              </span>
                            </div>
                            <div className="activity-details">
                              <span className="activity-time">
                                {new Date((activity.last_timestamp || 0) * 1000).toLocaleTimeString()}
                              </span>
                              {activity.confidence && (
                                <span className="activity-confidence">
                                  {Math.round(activity.confidence * 100)}% conf
                                </span>
                              )}
                              {activity.audio_quality && (
                                <span className={`audio-quality-badge ${activity.audio_quality}`}>
                                  {activity.audio_quality}
                                </span>
                              )}
                              {profile.speaking_pattern_score && (
                                <span className="pattern-score">
                                  Pattern: {Math.round(profile.speaking_pattern_score * 100)}%
                                </span>
                              )}
                            </div>
                          </div>
                        );
                      })}
                  </div>
                </div>
              )}

              {/* Enhanced Conversation Insights Panel */}
              {Object.keys(conversationInsights).length > 0 && (
                <div className="conversation-insights">
                  <h4>Advanced Conversation Analysis</h4>
                  <div className="insights-grid">
                    {conversationInsights.meeting_pace && (
                      <div className="insight">
                        <span className="insight-label">Pace:</span>
                        <span className={`insight-value pace-${conversationInsights.meeting_pace}`}>
                          {conversationInsights.meeting_pace}
                        </span>
                      </div>
                    )}
                    {conversationInsights.most_active_speaker && (
                      <div className="insight">
                        <span className="insight-label">Most Active:</span>
                        <span className="insight-value">{conversationInsights.most_active_speaker}</span>
                      </div>
                    )}
                    {conversationInsights.total_transitions !== undefined && (
                      <div className="insight">
                        <span className="insight-label">Transitions:</span>
                        <span className="insight-value">{conversationInsights.total_transitions}</span>
                      </div>
                    )}
                    {conversationInsights.speaker_dominance && (
                      <div className="insight dominance">
                        <span className="insight-label">Dominance:</span>
                        <div className="dominance-bars">
                          {Object.entries(conversationInsights.speaker_dominance)
                            .sort((a, b) => b[1] - a[1])
                            .slice(0, 3)
                            .map(([speaker, dominance]) => (
                              <div key={speaker} className="dominance-item">
                                <span className="dominance-speaker">{speaker}</span>
                                <div className="dominance-bar">
                                  <div 
                                    className="dominance-fill"
                                    style={{ width: `${dominance * 100}%` }}
                                  ></div>
                                </div>
                                <span className="dominance-percent">{Math.round(dominance * 100)}%</span>
                              </div>
                            ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
              
              {/* Speaker Overlap History */}
              {speakerOverlaps.length > 0 && (
                <div className="overlap-history">
                  <h4>Recent Speaker Overlaps</h4>
                  <div className="overlap-list">
                    {speakerOverlaps.slice(0, 5).map(overlap => (
                      <div key={overlap.id} className="overlap-item">
                        <span className="overlap-speakers">
                          {overlap.speakers.join(' & ')}
                        </span>
                        <span className="overlap-duration">
                          {overlap.duration.toFixed(1)}s
                        </span>
                        <span className="overlap-time">
                          {overlap.time}
                        </span>
                        <span className={`overlap-type ${overlap.type}`}>
                          {overlap.type}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          <div className="agenda-section">
            <h3>Meeting Agenda</h3>
            <textarea
              value={agenda}
              onChange={(e) => setAgenda(e.target.value)}
              placeholder="Enter agenda items (one per line)"
              className="agenda-input"
              rows="4"
            />
            <button onClick={setMeetingAgenda} className="set-agenda-button">
              Set Agenda
            </button>
            {agendaItems.length > 0 && (
              <div className="agenda-items">
                <h4>Current Agenda:</h4>
                <ul>
                  {agendaItems.map((item, index) => (
                    <li key={index}>{item}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>

        {/* Enhanced Multi-Speaker Transcription */}
        {isRecording && (
          <div className="transcription-section">
            <div className="transcription-header">
              <h3>📝 Enhanced Multi-Speaker Live Transcription</h3>
              <div className="transcription-stats">
                <span className="transcript-count">{transcriptions.length} messages</span>
                <span className="speakers-count">{new Set(transcriptions.map(t => t.speaker_id)).size} speakers</span>
                {conversationMomentum > 0 && (
                  <span className="momentum-indicator" style={{ color: getMomentumIndicator().color }}>
                    {getMomentumIndicator().icon} {getMomentumIndicator().text}
                  </span>
                )}
                {transcriptions.length > 0 && (
                  <button 
                    className="clear-transcripts"
                    onClick={() => setTranscriptions([])}
                  >
                    Clear
                  </button>
                )}
              </div>
            </div>
            <div className="transcription-container" ref={transcriptionRef}>
              {transcriptions.length === 0 ? (
                <div className="no-transcription">
                  <div className="waiting-indicator">
                    <div className="pulse-ring"></div>
                    <span>Advanced multi-speaker transcription active...</span>
                  </div>
                </div>
              ) : (
                transcriptions.map((transcription, index) => {
                  const isCurrentSpeaker = transcription.speaker_id === currentSpeaker;
                  const profile = speakerProfiles[transcription.speaker_id] || {};
                  
                  return (
                    <div 
                      key={transcription.id} 
                      className={`transcription-item ${isCurrentSpeaker ? 'current-speaker' : ''} ${transcription.is_question ? 'question' : ''}`}
                    >
                      <div className="transcription-header-item">
                        <span 
                          className="speaker-name"
                          style={{ 
                            color: getSpeakerStatusColor(transcription.speaker_id),
                            fontWeight: isCurrentSpeaker ? 'bold' : 'normal'
                          }}
                        >
                          {transcription.speaker_id}
                          {isCurrentSpeaker && <span className="live-indicator">🔴 LIVE</span>}
                        </span>
                        <div className="transcription-meta">
                          <span className="transcription-time">{transcription.time}</span>
                          {transcription.confidence && (
                            <span className={`confidence-score ${
                              transcription.confidence > 0.9 ? 'very-high' : 
                              transcription.confidence > 0.8 ? 'high' : 
                              transcription.confidence > 0.6 ? 'medium' : 'low'
                            }`}>
                              {Math.round(transcription.confidence * 100)}% conf
                            </span>
                          )}
                          {transcription.word_count > 0 && (
                            <span className="word-count">{transcription.word_count} words</span>
                          )}
                          {transcription.speaking_rate && (
                            <span className="speaking-rate">
                              {transcription.speaking_rate.toFixed(1)} wpm
                            </span>
                          )}
                          {profile.voice_consistency && (
                            <span className="voice-consistency">
                              Voice: {Math.round(profile.voice_consistency * 100)}%
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="transcription-text">
                        {transcription.text}
                        {transcription.is_question && <span className="question-indicator">❓</span>}
                      </div>
                      
                      {/* Enhanced context information */}
                      <div className="transcription-context">
                        {transcription.sentiment_score !== null && (
                          <span className={`sentiment-indicator ${
                            transcription.sentiment_score > 0.1 ? 'positive' : 
                            transcription.sentiment_score < -0.1 ? 'negative' : 'neutral'
                          }`}>
                            Tone: {
                              transcription.sentiment_score > 0.1 ? '😊 Positive' : 
                              transcription.sentiment_score < -0.1 ? '😔 Negative' : '😐 Neutral'
                            }
                          </span>
                        )}
                        {index === transcriptions.length - 1 && transcription.conversation_momentum !== undefined && (
                          <span className="momentum-context" style={{ color: getMomentumIndicator().color }}>
                            Momentum: {getMomentumIndicator().text}
                          </span>
                        )}
                        {transcription.text_length > 100 && (
                          <span className="length-indicator">Long statement</span>
                        )}
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        )}

        {/* Prompts Section */}
        {prompts.length > 0 && (
          <div className="prompts-section">
            <h3>📋 Facilitator Prompts</h3>
            {prompts.map((prompt, index) => (
              <div key={index} className={`prompt ${prompt.prompt_type}`}>
                <div className="prompt-content">
                  <strong>{prompt.prompt_type.replace('_', ' ').toUpperCase()}:</strong>
                  <p>{prompt.message}</p>
                </div>
                <button
                  onClick={() => dismissPrompt(index)}
                  className="dismiss-button"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Enhanced Multi-Speaker Visualization */}
        <div className="chart-section">
          <div className="chart-container">
            {speakerData.length > 0 ? (
              <div className="chart-with-stats">
                <div className="chart-wrapper">
                  <Bar data={chartData} options={chartOptions} />
                </div>
                <div className="speaker-stats fade-in">
                  <h3>📊 Enhanced Multi-Speaker Analytics</h3>
                  <div className="stats-grid">
                    {speakerData.map((speaker, index) => {
                      const profile = speakerProfiles[speaker.speaker_id] || {};
                      const isActive = activeSpeakers.has(speaker.speaker_id);
                      const isPrimary = speaker.speaker_id === currentSpeaker;
                      
                      return (
                        <div 
                          key={speaker.speaker_id} 
                          className={`speaker-stat ${isActive ? 'currently-speaking' : ''} ${isPrimary ? 'primary-speaker' : ''} ${speaker.speaker_id === dominantSpeaker ? 'dominant-speaker' : ''}`}
                          style={{
                            animationDelay: `${index * 0.1}s`,
                            borderColor: getSpeakerStatusColor(speaker.speaker_id)
                          }}
                        >
                          <div className="speaker-info">
                            <div className="speaker-name-container">
                              <span className="speaker-name">
                                {speaker.speaker_id}
                                {isActive && (
                                  <span className={`speaking-badge animate-pulse ${isPrimary ? 'primary' : 'secondary'}`}>
                                    🎤 {isPrimary ? 'Primary' : 'Active'}
                                  </span>
                                )}
                                {speaker.speaker_id === dominantSpeaker && (
                                  <span className="dominant-badge">
                                    👑 Most Active
                                  </span>
                                )}
                              </span>
                              <div className="speaker-details">
                                {speaker.speech_segments > 0 && (
                                  <span className="segment-count">
                                    {speaker.speech_segments} segments
                                  </span>
                                )}
                                {speaker.confidence_score && (
                                  <span className={`confidence-indicator ${
                                    speaker.confidence_score > 0.9 ? 'very-high' : 
                                    speaker.confidence_score > 0.8 ? 'high' : 
                                    speaker.confidence_score > 0.6 ? 'medium' : 'low'
                                  }`}>
                                    {Math.round(speaker.confidence_score * 100)}% ID confidence
                                  </span>
                                )}
                                {speaker.voice_consistency && (
                                  <span className="voice-consistency-indicator">
                                    Voice: {Math.round(speaker.voice_consistency * 100)}%
                                  </span>
                                )}
                              </div>
                            </div>
                            <div className="speaker-metrics">
                              <span className="speaker-time">
                                {formatTimeDetailed(speaker.talk_time_in_seconds)}
                              </span>
                              <span className="speaker-percentage">
                                {speaker.percentage.toFixed(1)}%
                              </span>
                              {speaker.avg_speaking_duration > 0 && (
                                <span className="avg-duration">
                                  Avg: {speaker.avg_speaking_duration.toFixed(1)}s
                                </span>
                              )}
                            </div>
                          </div>
                          <div className="progress-bar">
                            <div
                              className={`progress-fill ${isActive ? 'active-speaker' : ''} ${isPrimary ? 'primary' : ''}`}
                              style={{
                                width: `${Math.max(speaker.percentage, 2)}%`,
                                backgroundColor: chartData.datasets[0].backgroundColor[index],
                              }}
                            />
                          </div>
                          
                          {/* Enhanced speaker characteristics */}
                          <div className="speaker-characteristics">
                            {(speaker.avg_pitch || speaker.avg_energy) && (
                              <div className="audio-characteristics">
                                {speaker.avg_pitch && (
                                  <span className="pitch-indicator">
                                    🎵 {speaker.avg_pitch.toFixed(0)}Hz
                                  </span>
                                )}
                                {speaker.avg_energy && (
                                  <span className="energy-indicator">
                                    ⚡ {(speaker.avg_energy * 1000).toFixed(1)}
                                  </span>
                                )}
                              </div>
                            )}
                            
                            {/* Interaction metrics */}
                            <div className="interaction-metrics">
                              {speaker.interrupt_count > 0 && (
                                <span className="interrupt-metric negative">
                                  Interrupts: {speaker.interrupt_count}
                                </span>
                              )}
                              {speaker.interrupted_count > 0 && (
                                <span className="interrupted-metric">
                                  Interrupted: {speaker.interrupted_count}
                                </span>
                              )}
                              {speaker.speaking_pattern_score && (
                                <span className={`pattern-metric ${
                                  speaker.speaking_pattern_score > 0.7 ? 'consistent' : 
                                  speaker.speaking_pattern_score > 0.4 ? 'moderate' : 'varied'
                                }`}>
                                  Pattern: {Math.round(speaker.speaking_pattern_score * 100)}%
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  
                  {/* Enhanced meeting summary */}
                  <div className="meeting-summary">
                    <div className="summary-stat">
                      <span className="summary-label">Active Speaking</span>
                      <span className="summary-value">
                        {((totalSpeakingTime / Math.max(meetingDuration, 1)) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="summary-stat">
                      <span className="summary-label">Silence</span>
                      <span className="summary-value">
                        {((silenceTime / Math.max(meetingDuration, 1)) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="summary-stat">
                      <span className="summary-label">Participants</span>
                      <span className="summary-value">{speakerData.length}</span>
                    </div>
                    <div className="summary-stat">
                      <span className="summary-label">Currently Active</span>
                      <span className="summary-value">{activeSpeakers.size}</span>
                    </div>
                    {speakerTransitions > 0 && (
                      <div className="summary-stat">
                        <span className="summary-label">Speaker Changes</span>
                        <span className="summary-value">{speakerTransitions}</span>
                      </div>
                    )}
                    <div className="summary-stat">
                      <span className="summary-label">Conversation Energy</span>
                      <span className="summary-value" style={{ color: getMomentumIndicator().color }}>
                        {getMomentumIndicator().text}
                      </span>
                    </div>
                    {speakerOverlaps.length > 0 && (
                      <div className="summary-stat">
                        <span className="summary-label">Recent Overlaps</span>
                        <span className="summary-value">{speakerOverlaps.length}</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <div className="no-data">
                <h3>🎯 Ready for Advanced Multi-Speaker Analysis</h3>
                <p>
                  {isRecording
                    ? 'Advanced AI-powered multi-speaker detection is active. Multiple people can speak simultaneously and will be tracked with enhanced accuracy!'
                    : 'Click "Start Meeting" to begin enhanced real-time multi-speaker analysis with comprehensive voice profiling, overlap detection, and conversation flow tracking.'}
                </p>
                {!isRecording && (
                  <div className="feature-preview">
                    <div className="preview-item">⚡ Multi-level voice activity detection with overlap handling</div>
                    <div className="preview-item">🧠 Comprehensive voice feature extraction and speaker profiling</div>
                    <div className="preview-item">🔄 Advanced conversation flow and transition pattern learning</div>
                    <div className="preview-item">📊 Real-time confidence scoring and voice consistency tracking</div>
                    <div className="preview-item">💬 Conversation momentum and context-aware processing</div>
                    <div className="preview-item">👥 Simultaneous multi-speaker support with primary/secondary detection</div>
                    <div className="preview-item">🎯 Speaker overlap detection and interaction analysis</div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Enhanced Instructions */}
        <div className="instructions">
          <h3>📖 Advanced Multi-Speaker Features</h3>
          <div className="feature-list">
            <div className="feature enhanced">
              <span className="feature-icon">👥</span>
              <div className="feature-content">
                <strong>Simultaneous Multi-Speaker Support</strong>
                <span>Handles multiple people speaking at once with primary/secondary speaker detection</span>
              </div>
            </div>
            <div className="feature enhanced">
              <span className="feature-icon">⚡</span>
              <div className="feature-content">
                <strong>Advanced Voice Activity Detection</strong>
                <span>Multi-level VAD with WebRTC + energy-based backup for robust overlap handling</span>
              </div>
            </div>
            <div className="feature enhanced">
              <span className="feature-icon">🧠</span>
              <div className="feature-content">
                <strong>Comprehensive Voice Profiling</strong>
                <span>MFCC, spectral, chroma, and energy features with voice consistency tracking</span>
              </div>
            </div>
            <div className="feature enhanced">
              <span className="feature-icon">🔄</span>
              <div className="feature-content">
                <strong>Conversation Flow Intelligence</strong>
                <span>Learns speaker patterns, predicts transitions, and tracks conversation momentum</span>
              </div>
            </div>
            <div className="feature enhanced">
              <span className="feature-icon">📊</span>
              <div className="feature-content">
                <strong>Adaptive Speaker Recognition</strong>
                <span>Dynamic thresholds, confidence scoring, and voice consistency monitoring</span>
              </div>
            </div>
            <div className="feature enhanced">
              <span className="feature-icon">🎤</span>
              <div className="feature-content">
                <strong>Enhanced Transcription Quality</strong>
                <span>Improved preprocessing, confidence filtering, and multi-speaker transcription</span>
              </div>
            </div>
            <div className="feature enhanced">
              <span className="feature-icon">⚠️</span>
              <div className="feature-content">
                <strong>Speaker Overlap Detection</strong>
                <span>Real-time detection and analysis of simultaneous speech with duration tracking</span>
              </div>
            </div>
            <div className="feature enhanced">
              <span className="feature-icon">🎯</span>
              <div className="feature-content">
                <strong>Interaction Analysis</strong>
                <span>Tracks interruptions, speaker dominance, and conversation patterns</span>
              </div>
            </div>
          </div>
          
          <div className="version-info">
            <h4>🔧 Backend Version: {backendVersion}</h4>
            <p>
              This advanced version provides significantly enhanced multi-speaker capabilities including:
              simultaneous speaker detection, voice consistency tracking, speaker overlap analysis,
              and comprehensive conversation flow intelligence. The system adapts to conversation
              patterns and becomes more accurate over time.
            </p>
            {serverFeatures.length > 0 && (
              <div className="server-features">
                <strong>Active Features:</strong>
                <ul>
                  {serverFeatures.map((feature, index) => (
                    <li key={index}>{feature.replace(/_/g, ' ')}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;