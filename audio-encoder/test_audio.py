import sys
import os

# Add the directory to sys.path so we can import the engine
sys.path.append(os.path.join(os.path.dirname(__file__), 'audio_understanding'))

from audio_understanding_engine import VoiceUnderstandingEngine

def main():
    engine = VoiceUnderstandingEngine()
    print("Extracting text from audio...")
    
    # Path to audio and output file
    audio_path = r"c:\Users\Sam-tech\Desktop\Multimodal-RAG-Platform\audio-encoder\input\Adele - Hello (Lyrics).mp3"
    output_path = r"c:\Users\Sam-tech\Desktop\Multimodal-RAG-Platform\audio-encoder\input\placeholder.txt"
    
    result = engine.transcribe_with_timestamps(
        audio_input=audio_path
    )
    
    text = result.get('text', '')
    chunks = result.get('chunks', [])
    
    # Format the output text
    formatted_text = "Full Text:\n" + text + "\n\nSegments:\n"
    for chunk in chunks:
        if 'timestamp' in chunk and len(chunk['timestamp']) == 2:
            formatted_text += f"[{chunk['timestamp'][0]:.2f}s -> {chunk['timestamp'][1]:.2f}s] {chunk['text']}\n"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(formatted_text)
        
    print(f"Extraction complete! Text saved to {output_path}")

if __name__ == '__main__':
    main()
