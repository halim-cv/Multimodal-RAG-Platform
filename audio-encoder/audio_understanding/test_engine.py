from audio_understanding_engine import VoiceUnderstandingEngine
import warnings
warnings.filterwarnings("ignore")

engine = VoiceUnderstandingEngine()
try:
    print("Transcribing dummy audio (1 sec of zeros)...")
    import numpy as np
    audio = np.zeros(16000, dtype=np.float32)
    res = engine.transcribe_with_timestamps(audio)
    print("SUCCESS")
    print(res)
except Exception as e:
    import traceback
    traceback.print_exc()
