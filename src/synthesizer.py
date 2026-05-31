import os
import wave
from typing import List
import numpy as np
from music21 import pitch

def get_pitch_frequency(pitch_name: str) -> float:
    """
    Translates a pitch name (e.g. 'C4', 'E-4', 'F#4') to its corresponding frequency in Hz.
    Falls back to middle C (261.63 Hz) on error.
    """
    try:
        p = pitch.Pitch(pitch_name)
        return float(p.frequency)
    except Exception:
        # Fallback to Middle C
        return 261.63

def synthesize_token(token: str, bpm: int = 120, sample_rate: int = 44100) -> np.ndarray:
    """
    Synthesizes a single musical token (Note, Chord, or Rest) into a mono floating point audio array.
    Uses multi-harmonic synthesis and custom ADSR envelopes for a warm, premium sound.
    """
    try:
        parts = token.rsplit("_", 1)
        if len(parts) != 2:
            return np.array([], dtype=np.float32)
            
        element_str, duration_str = parts[0], parts[1]
        duration_quarters = float(duration_str)
        
        # Convert duration from beats (quarter notes) to seconds
        duration_seconds = duration_quarters * (60.0 / bpm)
        if duration_seconds <= 0:
            return np.array([], dtype=np.float32)
            
        num_samples = int(sample_rate * duration_seconds)
        t = np.linspace(0, duration_seconds, num_samples, endpoint=False)
        
        # 1. Handle Silence (Rests)
        if element_str == "rest":
            return np.zeros(num_samples, dtype=np.float32)
            
        # 2. Extract Pitch Frequencies
        frequencies = []
        if "." in element_str:
            # Chord pitches
            pitches_list = element_str.split(".")
            frequencies = [get_pitch_frequency(p) for p in pitches_list]
        else:
            # Single pitch
            frequencies = [get_pitch_frequency(element_str)]
            
        if not frequencies:
            return np.zeros(num_samples, dtype=np.float32)
            
        # 3. Synthesize Multi-Harmonic Waveforms to create a rich timbre
        # Fundamental (1x freq) + 2nd harmonic (2x freq) + 3rd harmonic (3x freq)
        wave_sum = np.zeros(num_samples, dtype=np.float32)
        for freq in frequencies:
            # Synthesize fundamental
            sine_fund = 0.60 * np.sin(2 * np.pi * freq * t)
            # Synthesize second harmonic (adds warm octave body)
            sine_2nd = 0.25 * np.sin(2 * np.pi * (2 * freq) * t)
            # Synthesize third harmonic (adds rich overtone color)
            sine_3rd = 0.15 * np.sin(2 * np.pi * (3 * freq) * t)
            
            wave_sum += (sine_fund + sine_2nd + sine_3rd)
            
        # Normalize sum of frequencies to prevent digital clipping
        wave_sum /= len(frequencies)
        
        # 4. Apply a professional Linear-Exponential ADSR Envelope
        # Fast linear fade-in (0.01 seconds) to avoid pops and clicks
        attack_samples = int(sample_rate * min(0.01, duration_seconds / 4.0))
        envelope = np.ones(num_samples, dtype=np.float32)
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0.0, 1.0, attack_samples)
            
        # Exponential decay envelope (decay constant scales with duration)
        decay_speed = max(2.5, 4.0 / duration_seconds)
        decay_envelope = np.exp(-decay_speed * t)
        
        # Combine envelopes
        wave_sum *= envelope * decay_envelope
        
        return wave_sum.astype(np.float32)
        
    except Exception as e:
        print(f"Warning: Synthesizer failed to render token '{token}'. Error: {e}")
        return np.array([], dtype=np.float32)

def generate_wav_file(
    generated_tokens: List[str], 
    output_path: str = "output/generated_music.wav", 
    bpm: int = 120, 
    sample_rate: int = 44100
):
    """
    Synthesizes a list of musical tokens into a stereo PCM 16-bit WAV file.
    Applies spatial delay on the right channel to create a spacious, professional stereo reverb effect.
    """
    print("Synthesizing generated music to CD-quality audio...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. Synthesize all tokens sequentially and concatenate
    audio_chunks = []
    for token in generated_tokens:
        chunk = synthesize_token(token, bpm=bpm, sample_rate=sample_rate)
        if len(chunk) > 0:
            audio_chunks.append(chunk)
            
    if not audio_chunks:
        print("Error: No valid audio chunks generated. WAV creation aborted.")
        return
        
    mono_signal = np.concatenate(audio_chunks)
    
    # Normalize peak amplitude to -1.0dB FS to ensure maximum volume without clipping
    peak = np.max(np.abs(mono_signal))
    if peak > 0:
        mono_signal = (mono_signal / peak) * 0.89  # 0.89 is roughly -1dB
        
    # 2. Add spatial stereo enhancement (Chorus / Stereo-Widener effect)
    # Right channel is delayed by 18ms and mixed with the left to create space
    delay_samples = int(sample_rate * 0.018)
    left_channel = mono_signal
    
    right_channel = np.zeros_like(mono_signal)
    if len(mono_signal) > delay_samples:
        # Shift with roll and blend to simulate room bounce
        right_channel = 0.85 * np.roll(mono_signal, delay_samples) + 0.15 * mono_signal
    else:
        right_channel = mono_signal
        
    # Convert to 16-bit signed integers (-32768 to 32767)
    left_pcm = (left_channel * 32767.0).astype(np.int16)
    right_pcm = (right_channel * 32767.0).astype(np.int16)
    
    # Interleave channels for Stereo WAV
    stereo_pcm = np.empty((2 * len(left_pcm),), dtype=np.int16)
    stereo_pcm[0::2] = left_pcm
    stereo_pcm[1::2] = right_pcm
    
    # 3. Write PCM frames using built-in wave module
    try:
        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(2)      # Stereo
            wav_file.setsampwidth(2)      # 16-bit (2 bytes per sample)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(stereo_pcm.tobytes())
            
        print(f"Synthesized WAV successfully saved to: {output_path}")
    except Exception as e:
        print(f"Error: Failed to write WAV file. {e}")
