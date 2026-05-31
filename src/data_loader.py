import os
import glob
from typing import List, Tuple
from music21 import converter, instrument, note, chord, corpus

def parse_midi_element(element) -> str:
    """
    Parses a single music21 element (Note, Chord, or Rest) and returns a string token 
    representing its pitch/combination and duration.
    
    Example formats:
    - Note: "C4_0.5" (pitch C4 with duration 0.5 quarter length)
    - Chord: "C4.E4.G4_1.0" (pitches C4, E4, G4 with duration 1.0)
    - Rest: "rest_0.25" (rest with duration 0.25)
    """
    duration = round(float(element.duration.quarterLength), 3)
    
    if isinstance(element, note.Note):
        return f"{element.pitch.nameWithOctave}_{duration}"
    elif isinstance(element, chord.Chord):
        # Sort pitches to ensure chord representation is deterministic
        pitches = sorted([p.nameWithOctave for p in element.pitches])
        chord_str = ".".join(pitches)
        return f"{chord_str}_{duration}"
    elif isinstance(element, note.Rest):
        return f"rest_{duration}"
    return ""

def parse_midi_file(file_path: str) -> List[str]:
    """
    Parses a single MIDI file and extracts sequential note, chord, and rest tokens.
    """
    try:
        # Load MIDI file
        midi_data = converter.parse(file_path)
        
        # Flatten and filter for Notes, Chords, and Rests
        elements = midi_data.flatten().notesAndRests
        
        tokens = []
        for element in elements:
            token = parse_midi_element(element)
            if token:
                tokens.append(token)
        return tokens
    except Exception as e:
        print(f"Warning: Failed to parse {file_path}. Error: {e}")
        return []

def load_midi_corpus(directory_path: str) -> List[str]:
    """
    Loads all MIDI files in the given directory and returns a flat list of tokens.
    If the directory is empty or doesn't exist, returns an empty list.
    """
    if not os.path.exists(directory_path):
        return []
        
    search_path = os.path.join(directory_path, "**", "*.mid*")
    midi_files = glob.glob(search_path, recursive=True)
    
    if not midi_files:
        return []
        
    print(f"Found {len(midi_files)} MIDI files in {directory_path}. Parsing...")
    all_tokens = []
    for file_path in midi_files:
        tokens = parse_midi_file(file_path)
        all_tokens.extend(tokens)
        
    print(f"Successfully loaded {len(all_tokens)} musical tokens from local dataset.")
    return all_tokens

def load_builtin_chorales(num_chorales: int = 15) -> List[str]:
    """
    Fallback loader that fetches a selection of Bach chorales from music21's built-in corpus.
    This ensures the pipeline is fully functional with zero setup/downloads.
    """
    print(f"Using built-in Bach Chorales corpus (loading {num_chorales} pieces)...")
    all_tokens = []
    
    try:
        # Load Bach chorales iterator
        chorale_iterator = corpus.chorales.Iterator()
        loaded_count = 0
        
        for chorale in chorale_iterator:
            if loaded_count >= num_chorales:
                break
                
            # Flatten and filter notes and rests
            elements = chorale.flatten().notesAndRests
            tokens = [parse_midi_element(el) for el in elements if parse_midi_element(el)]
            all_tokens.extend(tokens)
            loaded_count += 1
            
        print(f"Successfully loaded {len(all_tokens)} musical tokens from {loaded_count} Bach chorales.")
        return all_tokens
    except Exception as e:
        print(f"Error loading built-in chorales: {e}")
        return []

def get_musical_data(directory_path: str, fallback_num: int = 15) -> List[str]:
    """
    High-level entrypoint: loads from local MIDI folder if available, 
    otherwise falls back to built-in Bach chorales.
    """
    tokens = load_midi_corpus(directory_path)
    if not tokens:
        print(f"No MIDI files found in '{directory_path}'. Falling back to built-in corpus.")
        tokens = load_builtin_chorales(fallback_num)
    return tokens
