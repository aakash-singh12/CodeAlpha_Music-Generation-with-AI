import os
import random
from typing import List, Dict
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from music21 import stream, note, chord, instrument

def sample_with_temperature(prediction_probabilities: np.ndarray, temperature: float = 0.8) -> int:
    """
    Applies temperature-based scaling to prediction probabilities and samples an index.
    - Low temperature (< 0.5) makes the model conservative (loops or highly predictable).
    - High temperature (> 1.2) makes the model random and creative but potentially dissonant.
    - Balanced temperature (0.7 - 1.0) provides a sweet spot.
    """
    # Prevent division by zero and extreme values
    temperature = max(temperature, 0.01)
    
    # Apply temperature scaling in log-space (adding small epsilon to avoid log(0))
    predictions = np.asarray(prediction_probabilities).astype("float64")
    log_predictions = np.log(predictions + 1e-10) / temperature
    exp_predictions = np.exp(log_predictions)
    
    # Re-normalize into probability distribution
    probabilities = exp_predictions / np.sum(exp_predictions)
    
    # Sample index using the temperature-adjusted probabilities
    choices = range(len(probabilities))
    return np.random.choice(choices, p=probabilities)

def generate_notes(
    model_path: str,
    note_to_int: Dict[str, int],
    int_to_note: Dict[int, str],
    seq_length: int = 64,
    gen_length: int = 150,
    temperature: float = 0.8,
    seed_sequence: List[str] = None
) -> List[str]:
    """
    Generates a sequence of musical tokens using the trained LSTM model.
    If no seed sequence is provided, a random seed is synthesized from unique vocabulary notes.
    """
    print(f"Loading trained model from {model_path}...")
    model = load_model(model_path)
    
    # If no seed is provided, create a random valid seed sequence from the vocabulary keys
    vocab_keys = list(note_to_int.keys())
    if not seed_sequence or len(seed_sequence) < seq_length:
        print("No valid seed sequence provided. Synthesizing a random starting seed...")
        seed_sequence = [random.choice(vocab_keys) for _ in range(seq_length)]
    else:
        # Take the last seq_length elements
        seed_sequence = seed_sequence[-seq_length:]
        
    print(f"Starting seed: {seed_sequence[:5]} ... {seed_sequence[-5:]}")
    
    # Map seed to integers
    pattern = [note_to_int[token] if token in note_to_int else note_to_int[random.choice(vocab_keys)] 
               for token in seed_sequence]
    
    generated_tokens = []
    
    print(f"Generating {gen_length} notes with temperature {temperature}...")
    for note_index in range(gen_length):
        # Reshape input to (1, seq_length) for model prediction
        prediction_input = np.reshape(pattern, (1, len(pattern)))
        
        # Predict probability distribution
        prediction = model.predict(prediction_input, verbose=0)[0]
        
        # Sample with temperature
        sampled_int = sample_with_temperature(prediction, temperature)
        sampled_token = int_to_note[sampled_int]
        
        generated_tokens.append(sampled_token)
        
        # Shift sliding window: append new prediction, remove oldest note
        pattern.append(sampled_int)
        pattern = pattern[1:]
        
    print("Generation complete!")
    return generated_tokens

def tokens_to_midi(generated_tokens: List[str], output_path: str = "output/generated_music.mid") -> stream.Stream:
    """
    Converts a sequence of string tokens back into a music21 stream and writes it as a MIDI file.
    
    Token formats:
    - Rest: "rest_0.5" -> music21.note.Rest of duration 0.5
    - Chord: "C4.E4.G4_1.0" -> music21.chord.Chord of pitches C4,E4,G4, duration 1.0
    - Note: "E-4_0.25" -> music21.note.Note of pitch E-4, duration 0.25
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    output_stream = stream.Stream()
    output_stream.append(instrument.Piano())
    
    for token in generated_tokens:
        # Separate the note/chord component from the duration
        try:
            parts = token.rsplit("_", 1)
            if len(parts) != 2:
                continue
                
            element_str, duration_str = parts[0], parts[1]
            duration = float(duration_str)
            
            # 1. Parse Rests
            if element_str == "rest":
                r = note.Rest()
                r.duration.quarterLength = duration
                output_stream.append(r)
                
            # 2. Parse Chords
            elif "." in element_str:
                notes_in_chord = element_str.split(".")
                c = chord.Chord(notes_in_chord)
                c.duration.quarterLength = duration
                output_stream.append(c)
                
            # 3. Parse Single Notes
            else:
                n = note.Note(element_str)
                n.duration.quarterLength = duration
                output_stream.append(n)
                
        except Exception as e:
            # Skip invalid formatting errors silently to preserve stream structure
            continue
            
    # Write the music21 stream to a MIDI file
    output_stream.write("midi", fp=output_path)
    print(f"Generated MIDI successfully saved to: {output_path}")
    
    return output_stream
