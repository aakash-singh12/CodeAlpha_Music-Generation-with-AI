import os
import sys
import argparse
from src.data_loader import get_musical_data
from src.preprocessor import build_vocabulary, save_mappings, load_mappings, prepare_sequences
from src.model import build_lstm_model
from src.trainer import train_model, plot_training_loss
from src.generator import generate_notes, tokens_to_midi
from src.synthesizer import generate_wav_file

# Define target paths
MIDI_DIR = "data/midi_dataset"
MODELS_DIR = "models"
OUTPUT_DIR = "output"
MAPPINGS_PATH = os.path.join(MODELS_DIR, "mappings.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "music_lstm_model.keras")

# ANSI Color Codes for Premium Console Styling
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_CYAN = "\033[36m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_BLUE = "\033[34m"
C_MAGENTA = "\033[35m"

def print_header(title: str):
    """Prints a styled header for CLI sections."""
    print(f"\n{C_BOLD}{C_CYAN}=== {title.upper()} ==={C_RESET}")

def run_preprocessing(seq_length: int) -> tuple:
    """Loads and preprocesses MIDI data, builds vocab mappings, and saves them."""
    print_header("Data Preparation & Preprocessing")
    
    # 1. Ensure directories exist
    os.makedirs(MIDI_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 2. Load musical tokens (auto-fallback to Bach chorales if folder is empty)
    tokens = get_musical_data(MIDI_DIR, fallback_num=18)
    if not tokens:
        print(f"{C_YELLOW}Warning: No musical tokens loaded. Add MIDI files or check music21 install.{C_RESET}")
        return None, None, 0
        
    print(f"Total musical tokens extracted: {C_GREEN}{len(tokens)}{C_RESET}")
    
    # 3. Build vocabulary mappings
    note_to_int, int_to_note, vocab_size = build_vocabulary(tokens)
    print(f"Unique vocabulary size: {C_GREEN}{vocab_size}{C_RESET} unique note/chord/duration combinations.")
    
    # 4. Save mappings to pickle
    save_mappings(note_to_int, int_to_note, MODELS_DIR)
    
    # 5. Build training sequences (X, y)
    try:
        X, y = prepare_sequences(tokens, note_to_int, vocab_size, seq_length)
        return X, y, vocab_size
    except Exception as e:
        print(f"{C_YELLOW}Error during sequence preparation: {e}{C_RESET}")
        return None, None, 0

def run_training(X, y, vocab_size: int, seq_length: int, epochs: int, batch_size: int):
    """Builds and trains the Keras LSTM model, saving best checkpoints and plotting loss."""
    print_header("Model Construction & Training")
    
    if X is None or y is None or vocab_size == 0:
        print(f"{C_YELLOW}Error: Cannot train. Please run Preprocessing first.{C_RESET}")
        return
        
    # 1. Build Keras model
    model = build_lstm_model(vocab_size=vocab_size, seq_length=seq_length)
    
    # 2. Train the model
    history = train_model(model, X, y, epochs=epochs, batch_size=batch_size, save_dir=MODELS_DIR)
    
    # 3. Plot training loss curve
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    loss_plot_path = os.path.join(OUTPUT_DIR, "loss_plot.png")
    plot_training_loss(history, loss_plot_path)
    
    print(f"\n{C_GREEN}[SUCCESS] Model training and visualization completed successfully!{C_RESET}")

def run_generation(seq_length: int, gen_length: int, temperature: float, bpm: int):
    """Generates notes using trained model and exports output files in MIDI and WAV formats."""
    print_header("Music Generation & Synthesis")
    
    # 1. Check if model and mappings exist
    if not os.path.exists(MODEL_PATH):
        print(f"{C_YELLOW}Error: Trained model not found at {MODEL_PATH}. Please train a model first.{C_RESET}")
        return
    if not os.path.exists(MAPPINGS_PATH):
        print(f"{C_YELLOW}Error: Mappings file not found at {MAPPINGS_PATH}. Please run Preprocessing first.{C_RESET}")
        return
        
    # 2. Load mappings
    note_to_int, int_to_note, vocab_size = load_mappings(MAPPINGS_PATH)
    
    # 3. Generate note tokens
    # For seeding, try to load data to get a realistic seed sequence
    seed = None
    try:
        tokens = get_musical_data(MIDI_DIR, fallback_num=10)
        if len(tokens) >= seq_length:
            # Pick a random segment as seed
            start_idx = random.randint(0, len(tokens) - seq_length - 1)
            seed = tokens[start_idx : start_idx + seq_length]
    except Exception:
        pass # Will fall back to synthesized seed automatically in generate_notes
        
    generated_tokens = generate_notes(
        model_path=MODEL_PATH,
        note_to_int=note_to_int,
        int_to_note=int_to_note,
        seq_length=seq_length,
        gen_length=gen_length,
        temperature=temperature,
        seed_sequence=seed
    )
    
    # 4. Convert tokens to MIDI
    output_midi_path = os.path.join(OUTPUT_DIR, "generated_music.mid")
    tokens_to_midi(generated_tokens, output_midi_path)
    
    # 5. Synthesize to WAV
    output_wav_path = os.path.join(OUTPUT_DIR, "generated_music.wav")
    generate_wav_file(generated_tokens, output_wav_path, bpm=bpm)
    
    print(f"\n{C_GREEN}[SUCCESS] Music generation and rendering successfully completed!{C_RESET}")
    print(f"Generated MIDI: {C_BOLD}{output_midi_path}{C_RESET}")
    print(f"Generated WAV:  {C_BOLD}{output_wav_path}{C_RESET}")

def run_interactive_menu():
    """Renders the beautiful CLI menu dashboard for user interaction."""
    # Ensure standard directories exist
    os.makedirs(MIDI_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # In-memory variables to keep track of preprocessed data during a single session
    X, y, vocab_size = None, None, 0
    seq_length = 64
    
    while True:
        print(f"\n{C_BOLD}{C_MAGENTA}============================================")
        print("          AI MUSIC GENERATOR SYSTEM         ")
        print(f"============================================{C_RESET}")
        print(f"  1. Preprocess Dataset (MIDI Files or Bach)")
        print(f"  2. Build and Train Deep LSTM Model")
        print(f"  3. Generate New Music & Synthesize Audio")
        print(f"  4. Run Full Pipeline (End-to-End)")
        print(f"  5. Exit")
        print(f"{C_BOLD}{C_MAGENTA}============================================{C_RESET}")
        
        try:
            choice = input(f"Select an option (1-5): {C_BOLD}").strip()
            print(C_RESET, end="")
        except (KeyboardInterrupt, EOFError):
            print(f"\n{C_YELLOW}Exiting AI Music Generator. Goodbye!{C_RESET}")
            sys.exit(0)
            
        if choice == "1":
            try:
                seq_in = input("Enter sequence length (default 64): ").strip()
                seq_len = int(seq_in) if seq_in else 64
            except ValueError:
                seq_len = 64
            seq_length = seq_len
            X, y, vocab_size = run_preprocessing(seq_len)
            
        elif choice == "2":
            if X is None or y is None:
                # Attempt to run preprocessing automatically
                print(f"{C_YELLOW}No preprocessed data in memory. Running preprocessing...{C_RESET}")
                X, y, vocab_size = run_preprocessing(seq_length)
                
            if X is not None:
                try:
                    epochs_in = input("Enter epochs to train (default 15 for quick run): ").strip()
                    epochs = int(epochs_in) if epochs_in else 15
                    batch_in = input("Enter batch size (default 64): ").strip()
                    batch = int(batch_in) if batch_in else 64
                except ValueError:
                    epochs, batch = 15, 64
                run_training(X, y, vocab_size, seq_length, epochs, batch)
                
        elif choice == "3":
            try:
                gen_in = input("Enter notes to generate (default 150): ").strip()
                gen_len = int(gen_in) if gen_in else 150
                temp_in = input("Enter temperature (default 0.8, range 0.1-1.5): ").strip()
                temp = float(temp_in) if temp_in else 0.8
                bpm_in = input("Enter tempo in BPM (default 120): ").strip()
                bpm = int(bpm_in) if bpm_in else 120
            except ValueError:
                gen_len, temp, bpm = 150, 0.8, 120
            run_generation(seq_length, gen_len, temp, bpm)
            
        elif choice == "4":
            print(f"\n{C_CYAN}Running Full End-to-End Pipeline...{C_RESET}")
            # Preprocess
            seq_length = 64
            X, y, vocab_size = run_preprocessing(seq_length)
            if X is not None:
                # Train (low epoch for fast validation/safety)
                run_training(X, y, vocab_size, seq_length, epochs=12, batch_size=64)
                # Generate
                run_generation(seq_length, gen_length=150, temperature=0.8, bpm=120)
                
        elif choice == "5":
            print(f"{C_CYAN}Thank you for creating music with AI. Goodbye!{C_RESET}")
            break
        else:
            print(f"{C_YELLOW}Invalid choice. Please select 1 to 5.{C_RESET}")

def main():
    """Main program entrypoint."""
    # Set up argparse for non-interactive execution (important for automation/verification)
    parser = argparse.ArgumentParser(description="AI Music Generation System in Python")
    parser.add_argument("--mode", type=str, choices=["preprocess", "train", "generate", "pipeline", "interactive"], 
                        default="interactive", help="Pipeline action mode to execute")
    parser.add_argument("--seq-len", type=int, default=64, help="Sliding window sequence length")
    parser.add_argument("--gen-len", type=int, default=150, help="Number of notes/chords to generate")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--temp", type=float, default=0.8, help="Creativity temperature (0.1 to 1.5)")
    parser.add_argument("--bpm", type=int, default=120, help="Tempo of synthesized WAV audio")
    
    args = parser.parse_args()
    
    # Fix console colors for standard Windows terminal by running a shell color command if needed
    os.system("") 
    
    import random # imported here for random seed splits
    
    if args.mode == "interactive":
        run_interactive_menu()
    elif args.mode == "preprocess":
        run_preprocessing(args.seq_len)
    elif args.mode == "train":
        X, y, vocab_size = run_preprocessing(args.seq_len)
        if X is not None:
            run_training(X, y, vocab_size, args.seq_len, args.epochs, args.batch_size)
    elif args.mode == "generate":
        run_generation(args.seq_len, args.gen_len, args.temp, args.bpm)
    elif args.mode == "pipeline":
        X, y, vocab_size = run_preprocessing(args.seq_len)
        if X is not None:
            run_training(X, y, vocab_size, args.seq_len, args.epochs, args.batch_size)
            run_generation(args.seq_len, args.gen_len, args.temp, args.bpm)

if __name__ == "__main__":
    main()
