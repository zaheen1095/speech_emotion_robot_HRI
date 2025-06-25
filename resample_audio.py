import os
import librosa
import soundfile as sf
from config import RAW_AUDIO_DIR, RESAMPLED_DIR, FEATURE_SETTINGS

def resample_wav(input_file, output_file):
    try:
        audio, original_sr = librosa.load(input_file, sr=None)
        target_sample_rate = FEATURE_SETTINGS['sample_rate']
        if original_sr != target_sample_rate:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sample_rate)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        sf.write(output_file, audio, target_sample_rate)
        print(f"Resampled and saved: {output_file}")
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

def process_split(split):
    for emotion in os.listdir(os.path.join(RAW_AUDIO_DIR, split)):
        emotion_dir = os.path.join(RAW_AUDIO_DIR, split, emotion)
        for filename in os.listdir(emotion_dir):
            input_path = os.path.join(emotion_dir, filename)
            output_path = os.path.join(RESAMPLED_DIR, split, emotion, filename)
            resample_wav(input_path, output_path)

if __name__ == "__main__":
    print("Starting resampling process...")
    process_split("train")
    process_split("test")
    print("Resampling completed.")
