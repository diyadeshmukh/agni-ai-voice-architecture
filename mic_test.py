"""
Agni AI - Microphone Audio Test

Records audio from the selected microphone and saves it
as a WAV file in the format expected by the STT service.

Audio format:
    Sample rate: 16 kHz
    Channels: 1 (mono)
    Encoding: PCM 16-bit
"""

import sounddevice as sd
from scipy.io.wavfile import write


# Audio configuration
SAMPLE_RATE = 16_000
DURATION = 10
CHANNELS = 1

# Microphone device identified on this system.
DEVICE = 1

OUTPUT_FILE = "mic_test.wav"


print("Recording started...")
print("Speak into your microphone.")
print("Recording for 10 seconds...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="int16",
    device=DEVICE,
)

sd.wait()

write(
    OUTPUT_FILE,
    SAMPLE_RATE,
    audio,
)

print("Recording finished.")
print(f"Saved as {OUTPUT_FILE}")