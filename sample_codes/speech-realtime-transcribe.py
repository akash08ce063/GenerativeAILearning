import pyaudio
import numpy as np
import time
import webrtcvad
from faster_whisper import WhisperModel
import threading

# Initialize Whisper Model
model = WhisperModel("base", device="cpu", compute_type="int8")

SAMPLE_RATE = 16000  # 16 kHz
FRAME_DURATION_MS = 20  # ✅ Valid frame size for VAD (10, 20, or 30ms)
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # ✅ Frame size in samples (320 samples for 20ms)
FRAME_BYTES = FRAME_SIZE * 2  # ✅ Frame size in bytes (16-bit PCM = 2 bytes per sample)

# Create a VAD instance
vad = webrtcvad.Vad(1)  # Mode 1 (balanced detection)

# Store speech frames
audio_frames = []
speech_start = False
silence_duration = 0.5  # Silence threshold (seconds)

def stream_audio():
    global audio_frames, speech_start

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=FRAME_SIZE)

    last_speech_time = None

    while True:
        try:
            audio_chunk = stream.read(FRAME_SIZE, exception_on_overflow=False)
            
            if len(audio_chunk) != FRAME_BYTES:
                print(f"⚠️ Skipping invalid frame (size {len(audio_chunk)} instead of {FRAME_BYTES})")
                continue  # Skip corrupt/partial frames

            audio_data = np.frombuffer(audio_chunk, dtype=np.int16)

            if detect_speech(audio_chunk):  # ✅ Pass raw bytes instead of np array
                if not speech_start:
                    print("🎙️ Speech started!")
                    speech_start = True
                    audio_frames = []  # Clear buffer for new speech

                last_speech_time = time.time()
                audio_frames.append(audio_data)

            else:
                if speech_start and last_speech_time and time.time() - last_speech_time > silence_duration:
                    print("🛑 Speech ended. Transcribing...")
                    transcribe_speech(audio_frames)
                    audio_frames = []  # Clear buffer after transcription
                    speech_start = False

        except Exception as e:
            print(f"⚠️ Error while processing frame: {e}")

def detect_speech(audio_bytes):
    """ Use WebRTC VAD to detect speech presence. """
    try:
        return vad.is_speech(audio_bytes, SAMPLE_RATE)
    except Exception as e:
        print(f"⚠️ VAD Error: {e}")
        return False  # Default to no speech detected if error occurs

def transcribe_speech(audio_frames):
    """ Convert accumulated audio to Whisper-compatible format and transcribe. """
    if not audio_frames:
        return  # Skip empty buffers

    audio = np.concatenate(audio_frames, axis=0)  # Merge frames
    audio = audio.astype(np.float32) / 32768.0  # Normalize to [-1, 1]

    # Transcribe the entire speech segment
    segments, _ = model.transcribe(audio, language="en", beam_size=5)

    # Print the transcription
    full_text = " ".join(segment.text for segment in segments)
    print(f"📝 Transcription: {full_text}")

# Start audio streaming in a separate thread
audio_thread = threading.Thread(target=stream_audio)
audio_thread.daemon = True
audio_thread.start()

# Keep the program running
while True:
    time.sleep(1)
