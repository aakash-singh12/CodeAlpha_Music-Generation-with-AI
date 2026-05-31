import os
import pickle
from typing import List, Tuple, Dict
import numpy as np

def build_vocabulary(tokens: List[str]) -> Tuple[Dict[str, int], Dict[int, str], int]:
    """
    Builds bidirectional mapping dictionaries for the unique tokens in the dataset.
    """
    unique_tokens = sorted(list(set(tokens)))
    vocab_size = len(unique_tokens)
    
    note_to_int = {token: number for number, token in enumerate(unique_tokens)}
    int_to_note = {number: token for number, token in enumerate(unique_tokens)}
    
    return note_to_int, int_to_note, vocab_size

def save_mappings(note_to_int: Dict[str, int], int_to_note: Dict[int, str], save_dir: str):
    """
    Saves the integer mapping dictionaries as a pickle file.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "mappings.pkl")
    with open(save_path, "wb") as filepath:
        pickle.dump((note_to_int, int_to_note), filepath)
    print(f"Mappings successfully saved to {save_path}")

def load_mappings(mappings_path: str) -> Tuple[Dict[str, int], Dict[int, str], int]:
    """
    Loads mapping dictionaries from a pickle file and returns them along with vocabulary size.
    """
    if not os.path.exists(mappings_path):
        raise FileNotFoundError(f"Mappings file not found at {mappings_path}")
        
    with open(mappings_path, "rb") as filepath:
        note_to_int, int_to_note = pickle.load(filepath)
    
    vocab_size = len(note_to_int)
    return note_to_int, int_to_note, vocab_size

def prepare_sequences(tokens: List[str], note_to_int: Dict[str, int], vocab_size: int, seq_length: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares input-output training pairs using a sliding window.
    
    Inputs:
    - Shape: (num_sequences, seq_length)
    - Value: Integer-encoded musical tokens.
    
    Outputs:
    - Shape: (num_sequences, vocab_size)
    - Value: One-hot encoded target token (next note).
    """
    network_input = []
    network_output = []
    
    # Create input-output pairs
    for i in range(0, len(tokens) - seq_length):
        sequence_in = tokens[i : i + seq_length]
        sequence_out = tokens[i + seq_length]
        
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
        
    n_patterns = len(network_input)
    
    if n_patterns == 0:
        raise ValueError(
            f"Dataset too small! Total tokens: {len(tokens)}, "
            f"but sequence length is set to {seq_length}. "
            "Please add more MIDI files or reduce sequence length."
        )
        
    # Reshape input for LSTM (samples, sequence_length)
    # Note: Keras Embedding layer expects shape (samples, sequence_length)
    X = np.array(network_input)
    
    # One-hot encode the output targets
    y = np.eye(vocab_size)[network_output]
    
    print(f"Prepared {n_patterns} training sequences.")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y
