import socket
import json
import struct
import numpy as np
import soundfile as sf
import time

# --- CONFIGURATION ---
ROBOT_IP = "192.168.0.3"   # <--- Updated to match your log
ROBOT_PORT = 7878

def get_audio_from_pepper():
    print(f"[1] Connecting to {ROBOT_IP}:{ROBOT_PORT}...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ROBOT_IP, ROBOT_PORT))
    print("[2] Connected. Sending RECORD command (5 seconds)...")

    # Clear buffer
    s.setblocking(0)
    try:
        while s.recv(1024): pass
    except: pass
    s.setblocking(1)

    # Request Record
    msg = json.dumps({"command": "record", "content": 5}).encode('utf-8')
    s.sendall(msg)

    # Receive Header (4 bytes length)
    print("[3] Waiting for audio data...")
    header_data = b""
    while len(header_data) < 4:
        chunk = s.recv(4 - len(header_data))
        if not chunk: raise Exception("Socket closed prematurely")
        header_data += chunk
    
    data_len = struct.unpack('>I', header_data)[0]
    print(f"[4] Receiving {data_len} bytes of audio...")

    # Receive Body
    raw_data = b""
    while len(raw_data) < data_len:
        chunk = s.recv(4096)
        if not chunk: break
        raw_data += chunk

    print("[5] Download complete.")
    s.close()
    return raw_data

def save_and_analyze(raw_data):
    # Convert raw bytes to numpy array
    audio_int16 = np.frombuffer(raw_data, dtype=np.int16)
    
    # Check if empty
    if len(audio_int16) == 0:
        print("\n[ERROR] Received 0 bytes of audio! The robot sent nothing.")
        return

    # Normalize to float
    audio_float = audio_int16.astype(np.float32) / 32768.0
    
    # Calculate Volume
    peak = np.max(np.abs(audio_float))
    rms = np.sqrt(np.mean(audio_float**2))
    
    print(f"\n--- AUDIO DIAGNOSTICS ---")
    print(f"Total Samples: {len(audio_float)}")
    print(f"Peak Volume:   {peak:.6f} (Max is 1.0)")
    print(f"Average Vol:   {rms:.6f}")
    
    # Save file
    filename = "test_pepper.wav"
    sf.write(filename, audio_float, 16000)
    print(f"Saved to:      {filename}")
    print("-------------------------\n")

    if peak < 0.001:
        print("❌ CONCLUSION: The audio is SILENT. Increasing Python volume won't help.")
        print("   -> Check Pepper's microphone settings or hardware.")
    else:
        print("✅ CONCLUSION: The audio is GOOD. The problem is in the AI/Whisper code.")

if __name__ == "__main__":
    try:
        raw = get_audio_from_pepper()
        save_and_analyze(raw)
        print("Done. Please open 'test_pepper.wav' and listen to it.")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")