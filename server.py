import os
import random
from flask import Flask, jsonify, request, send_file, render_template_string
from src.data_loader import get_musical_data
from src.preprocessor import load_mappings
from src.generator import generate_notes, tokens_to_midi
from src.synthesizer import generate_wav_file

app = Flask(__name__)

# Constants
MIDI_DIR = "data/midi_dataset"
MODELS_DIR = "models"
OUTPUT_DIR = "output"
MAPPINGS_PATH = os.path.join(MODELS_DIR, "mappings.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "music_lstm_model.keras")

@app.route("/")
def index():
    """Serves the main beautiful HTML5 client page."""
    # We will embed the HTML directly as a template string to keep deployment single-file and robust!
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/status")
def status():
    """Returns the current training and file status of the AI system."""
    has_model = os.path.exists(MODEL_PATH)
    has_mappings = os.path.exists(MAPPINGS_PATH)
    has_wav = os.path.exists(os.path.join(OUTPUT_DIR, "generated_music.wav"))
    has_loss_plot = os.path.exists(os.path.join(OUTPUT_DIR, "loss_plot.png"))
    
    vocab_size = 0
    unique_tokens = []
    if has_mappings:
        try:
            _, int_to_note, vocab_size = load_mappings(MAPPINGS_PATH)
            # Sample up to 10 tokens for preview
            unique_tokens = list(int_to_note.values())[:15]
        except Exception:
            pass
            
    return jsonify({
        "has_model": has_model,
        "has_mappings": has_mappings,
        "has_wav": has_wav,
        "has_loss_plot": has_loss_plot,
        "vocab_size": vocab_size,
        "vocab_preview": unique_tokens
    })

@app.route("/api/generate", methods=["POST"])
def generate():
    """API endpoint to trigger music generation dynamically."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(MAPPINGS_PATH):
        return jsonify({"success": False, "error": "Model or mappings not trained yet. Please run training first."}), 400
        
    try:
        data = request.get_json() or {}
        gen_len = int(data.get("gen_len", 150))
        temperature = float(data.get("temperature", 0.8))
        bpm = int(data.get("bpm", 120))
        seq_length = 16 # Align with the fast verification sequence length, or auto-detect
        
        # Load mappings
        note_to_int, int_to_note, vocab_size = load_mappings(MAPPINGS_PATH)
        
        # Get random seed sequence from training data if available
        seed = None
        try:
            tokens = get_musical_data(MIDI_DIR, fallback_num=10)
            if len(tokens) >= seq_length:
                start_idx = random.randint(0, len(tokens) - seq_length - 1)
                seed = tokens[start_idx : start_idx + seq_length]
        except Exception:
            pass
            
        # Generate note tokens
        generated_tokens = generate_notes(
            model_path=MODEL_PATH,
            note_to_int=note_to_int,
            int_to_note=int_to_note,
            seq_length=seq_length,
            gen_length=gen_len,
            temperature=temperature,
            seed_sequence=seed
        )
        
        # Convert to MIDI
        output_midi_path = os.path.join(OUTPUT_DIR, "generated_music.mid")
        tokens_to_midi(generated_tokens, output_midi_path)
        
        # Synthesize to WAV
        output_wav_path = os.path.join(OUTPUT_DIR, "generated_music.wav")
        generate_wav_file(generated_tokens, output_wav_path, bpm=bpm)
        
        return jsonify({
            "success": True, 
            "midi_path": "/audio/generated_music.mid", 
            "wav_path": "/audio/generated_music.wav"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/audio/generated_music.wav")
def get_wav():
    """Serves the generated WAV audio file without client-side caching."""
    wav_path = os.path.join(OUTPUT_DIR, "generated_music.wav")
    if os.path.exists(wav_path):
        resp = send_file(wav_path, mimetype="audio/wav")
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp
    return "WAV file not generated yet", 404

@app.route("/audio/generated_music.mid")
def get_midi():
    """Serves the generated MIDI file for download."""
    midi_path = os.path.join(OUTPUT_DIR, "generated_music.mid")
    if os.path.exists(midi_path):
        return send_file(midi_path, mimetype="audio/midi", as_attachment=True, download_name="generated_ai_music.mid")
    return "MIDI file not generated yet", 404

@app.route("/images/loss_plot.png")
def get_loss_plot():
    """Serves the training loss visualization plot."""
    plot_path = os.path.join(OUTPUT_DIR, "loss_plot.png")
    if os.path.exists(plot_path):
        return send_file(plot_path, mimetype="image/png")
    return "Loss plot not generated yet", 404

# HTML/CSS/JS frontend embedded natively
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Symphony AI - Interactive Music Studio</title>
    <!-- Modern Premium Typography -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
    
    <style>
        :root {
            --bg-dark: #09090e;
            --panel-bg: rgba(22, 22, 38, 0.45);
            --border-glow: rgba(139, 92, 246, 0.25);
            --primary: #8b5cf6;
            --primary-glow: rgba(139, 92, 246, 0.6);
            --secondary: #06b6d4;
            --secondary-glow: rgba(6, 182, 212, 0.6);
            --text-light: #f3f4f6;
            --text-muted: #9ca3af;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: radial-gradient(circle at 50% 0%, #1e1145 0%, var(--bg-dark) 70%);
            color: var(--text-light);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            overflow-x: hidden;
        }

        /* Glassmorphism Header */
        header {
            width: 100%;
            padding: 24px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        header h1 {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 1.8rem;
            background: linear-gradient(135deg, #a78bfa 0%, #22d3ee 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .studio-badge {
            background: rgba(139, 92, 246, 0.15);
            color: #c084fc;
            padding: 4px 10px;
            border-radius: 99px;
            font-size: 0.75rem;
            font-weight: bold;
            border: 1px solid rgba(139, 92, 246, 0.3);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Hero Layout */
        .container {
            width: 100%;
            max-width: 1280px;
            padding: 40px 24px;
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 32px;
            flex-grow: 1;
        }

        @media (max-width: 968px) {
            .container {
                grid-template-columns: 1fr;
            }
        }

        /* Premium Panels */
        .panel {
            background: var(--panel-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border-glow);
            border-radius: 24px;
            padding: 36px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
            transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            position: relative;
            overflow: hidden;
        }

        .panel::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            opacity: 0.8;
        }

        .panel:hover {
            border-color: rgba(6, 182, 212, 0.4);
            box-shadow: 0 30px 60px rgba(6, 182, 212, 0.1);
            transform: translateY(-2px);
        }

        .panel-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 24px;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        /* Interactive Parameters Controls */
        .control-group {
            margin-bottom: 24px;
        }

        .control-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            font-weight: 600;
            font-size: 0.95rem;
            color: var(--text-light);
        }

        .control-value {
            color: var(--secondary);
            font-family: 'Space Grotesk', sans-serif;
            font-weight: bold;
        }

        .slider {
            -webkit-appearance: none;
            width: 100%;
            height: 8px;
            border-radius: 99px;
            background: #1f1f35;
            outline: none;
            transition: background 0.2s;
        }

        .slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            cursor: pointer;
            box-shadow: 0 0 10px var(--primary-glow);
            transition: transform 0.1s;
        }

        .slider::-webkit-slider-thumb:hover {
            transform: scale(1.2);
        }

        /* Glowing Action Button */
        .btn-generate {
            display: block;
            width: 100%;
            padding: 18px;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: #fff;
            border: none;
            border-radius: 16px;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.15rem;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            box-shadow: 0 8px 24px var(--primary-glow);
            position: relative;
            overflow: hidden;
            letter-spacing: 0.5px;
        }

        .btn-generate:hover {
            transform: scale(1.02);
            box-shadow: 0 12px 36px var(--secondary-glow);
        }

        .btn-generate:active {
            transform: scale(0.98);
        }

        .btn-generate .pulse-circle {
            position: absolute;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.2);
            width: 100px;
            height: 100px;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%) scale(0);
            animation: pulse-effect 2s infinite ease-out;
            pointer-events: none;
        }

        @keyframes pulse-effect {
            0% { transform: translate(-50%, -50%) scale(0.2); opacity: 0.8; }
            100% { transform: translate(-50%, -50%) scale(3.5); opacity: 0; }
        }

        /* Custom Premium Web Player */
        .player-card {
            background: rgba(10, 10, 18, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 24px;
            margin-top: 32px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .player-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .player-info {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .track-title {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 1.1rem;
            color: #fff;
        }

        .track-artist {
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        .player-controls {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .btn-play {
            background: #fff;
            color: var(--bg-dark);
            border: none;
            width: 52px;
            height: 52px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 1.3rem;
            box-shadow: 0 4px 12px rgba(255, 255, 255, 0.3);
            transition: all 0.2s ease;
        }

        .btn-play:hover {
            transform: scale(1.08);
            box-shadow: 0 6px 18px rgba(255, 255, 255, 0.5);
        }

        /* Dynamic Web Audio Visualizer */
        .visualizer-container {
            width: 100%;
            height: 80px;
            background: #06060c;
            border-radius: 12px;
            overflow: hidden;
            position: relative;
            border: 1px solid rgba(255, 255, 255, 0.03);
        }

        #visualizerCanvas {
            width: 100%;
            height: 100%;
        }

        .download-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-top: 16px;
        }

        .btn-download {
            padding: 12px;
            border-radius: 12px;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
            text-align: center;
            text-decoration: none;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 0.9rem;
        }

        .btn-download.wav {
            background: rgba(6, 182, 212, 0.1);
            color: var(--secondary);
            border: 1px solid rgba(6, 182, 212, 0.3);
        }

        .btn-download.wav:hover {
            background: var(--secondary);
            color: #fff;
        }

        .btn-download.mid {
            background: rgba(139, 92, 246, 0.1);
            color: var(--primary);
            border: 1px solid rgba(139, 92, 246, 0.3);
        }

        .btn-download.mid:hover {
            background: var(--primary);
            color: #fff;
        }

        /* Model Diagnostic & Status Sidebar */
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 32px;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 24px;
        }

        .stat-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 16px;
            text-align: center;
        }

        .stat-val {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.6rem;
            font-weight: 700;
            color: var(--secondary);
        }

        .stat-lbl {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 4px;
        }

        /* Loss Plot Image */
        .loss-plot-container {
            width: 100%;
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.05);
            margin-top: 16px;
            background: #09090e;
        }

        .loss-plot-img {
            width: 100%;
            display: block;
            transition: transform 0.3s;
        }

        .loss-plot-img:hover {
            transform: scale(1.02);
        }

        /* Vocabulary Chips */
        .vocab-container {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 12px;
            max-height: 120px;
            overflow-y: auto;
            padding-right: 4px;
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 99px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.3);
        }

        .vocab-chip {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 4px 10px;
            font-size: 0.8rem;
            color: #d1d5db;
            font-family: monospace;
        }

        /* Loading Spinner */
        .loader-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(9, 9, 14, 0.85);
            backdrop-filter: blur(8px);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 16px;
            z-index: 10;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s ease;
        }

        .loader-overlay.active {
            opacity: 1;
            pointer-events: auto;
        }

        .spinner {
            width: 48px;
            height: 48px;
            border: 4px solid rgba(6, 182, 212, 0.2);
            border-top: 4px solid var(--secondary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .loading-text {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: bold;
            letter-spacing: 0.5px;
            color: #fff;
        }
    </style>
</head>
<body>

    <header>
        <h1>Symphony AI <span class="studio-badge">Studio</span></h1>
        <div style="font-size: 0.9rem; color: var(--text-muted); display: flex; align-items: center; gap: 8px;">
            <div style="width: 8px; height: 8px; border-radius: 50%; background: #10b981; box-shadow: 0 0 8px #10b981;"></div>
            Active Session (Localhost)
        </div>
    </header>

    <div class="container">
        <!-- Main Interactive Studio -->
        <main class="panel" style="position: relative;">
            <div class="loader-overlay" id="loaderOverlay">
                <div class="spinner"></div>
                <div class="loading-text" id="loadingText">Generating AI Masterpiece...</div>
            </div>

            <h2 class="panel-title">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18V5l12-2v13"></path><circle cx="6" cy="18" r="3"></circle><circle cx="18" cy="16" r="3"></circle></svg>
                AI Generation Studio
            </h2>

            <div class="control-group">
                <div class="control-label">
                    <span>Creativity (Temperature)</span>
                    <span class="control-value" id="valTemp">0.8</span>
                </div>
                <input type="range" class="slider" id="paramTemp" min="0.2" max="1.5" step="0.05" value="0.8">
                <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 6px;">
                    Lower temperature yields standard melodies. Higher temperature increases creativity and harmonic variation.
                </div>
            </div>

            <div class="control-group">
                <div class="control-label">
                    <span>Symphony Length (Notes)</span>
                    <span class="control-value" id="valLen">150</span>
                </div>
                <input type="range" class="slider" id="paramLen" min="50" max="300" step="10" value="150">
            </div>

            <div class="control-group">
                <div class="control-label">
                    <span>Playback Tempo (BPM)</span>
                    <span class="control-value" id="valBPM">120</span>
                </div>
                <input type="range" class="slider" id="paramBPM" min="60" max="180" step="5" value="120">
            </div>

            <button class="btn-generate" id="btnGenerate">
                <div class="pulse-circle"></div>
                Compose AI Symphony
            </button>

            <!-- Custom Web Audio Player -->
            <div class="player-card">
                <div class="player-header">
                    <div class="player-info">
                        <span class="track-title" id="trackTitle">AI Generated Symphony</span>
                        <span class="track-artist">Deep LSTM Neural Network & Pure Synth Engine</span>
                    </div>
                    <div class="player-controls">
                        <button class="btn-play" id="btnPlay" disabled>▶</button>
                    </div>
                </div>
                
                <div class="visualizer-container">
                    <canvas id="visualizerCanvas"></canvas>
                </div>

                <div class="download-row">
                    <a class="btn-download wav" id="dlWav" href="#" style="pointer-events: none; opacity: 0.5;">Download WAV</a>
                    <a class="btn-download mid" id="dlMidi" href="#" style="pointer-events: none; opacity: 0.5;">Download MIDI</a>
                </div>
            </div>
        </main>

        <!-- Sidebar / Statistics -->
        <div class="sidebar">
            <!-- System Status Panel -->
            <div class="panel">
                <h2 class="panel-title">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="9" y1="9" x2="15" y2="15"></line><line x1="15" y1="9" x2="9" y2="15"></line></svg>
                    Neural Engine Status
                </h2>

                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-val" id="statModel">Offline</div>
                        <div class="stat-lbl">LSTM Model</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-val" id="statVocab">0</div>
                        <div class="stat-lbl">Vocab Size</div>
                    </div>
                </div>

                <div>
                    <span style="font-size: 0.9rem; font-weight: 600; color: #fff;">Vocabulary Tokens Sample</span>
                    <div class="vocab-container" id="vocabPreview">
                        <div style="font-size: 0.8rem; color: var(--text-muted); italic: true;">No mappings loaded</div>
                    </div>
                </div>
            </div>

            <!-- Loss Curve Visualization -->
            <div class="panel">
                <h2 class="panel-title">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18"></path><path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3"></path></svg>
                    Loss Curve Analysis
                </h2>
                <div style="font-size: 0.85rem; color: var(--text-muted);">
                    Categorical cross-entropy training loss profile saved by trainer callbacks:
                </div>
                <div class="loss-plot-container" id="plotContainer">
                    <div style="padding: 40px; text-align: center; color: var(--text-muted); font-size: 0.85rem;" id="noPlotText">
                        Loss plot not generated yet
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Web Audio Player and Visualizer Logic -->
    <script>
        const btnGenerate = document.getElementById("btnGenerate");
        const loaderOverlay = document.getElementById("loaderOverlay");
        const paramTemp = document.getElementById("paramTemp");
        const valTemp = document.getElementById("valTemp");
        const paramLen = document.getElementById("paramLen");
        const valLen = document.getElementById("valLen");
        const paramBPM = document.getElementById("paramBPM");
        const valBPM = document.getElementById("valBPM");
        
        const btnPlay = document.getElementById("btnPlay");
        const dlWav = document.getElementById("dlWav");
        const dlMidi = document.getElementById("dlMidi");
        
        const statModel = document.getElementById("statModel");
        const statVocab = document.getElementById("statVocab");
        const vocabPreview = document.getElementById("vocabPreview");
        
        let audioContext = null;
        let audioSource = null;
        let analyser = null;
        let audioBuffer = null;
        let isPlaying = false;
        let startTime = 0;
        let pausedAt = 0;
        
        // Update slider values dynamically
        paramTemp.addEventListener("input", (e) => valTemp.textContent = e.target.value);
        paramLen.addEventListener("input", (e) => valLen.textContent = e.target.value);
        paramBPM.addEventListener("input", (e) => valBPM.textContent = e.target.value);
        
        // Fetch engine status on page load
        async function fetchStatus() {
            try {
                const res = await fetch("/api/status");
                const data = await res.json();
                
                if (data.has_model) {
                    statModel.textContent = "Online";
                    statModel.style.color = "#10b981";
                } else {
                    statModel.textContent = "Missing";
                    statModel.style.color = "#f43f5e";
                }
                
                statVocab.textContent = data.vocab_size;
                
                // Load Vocab preview
                if (data.vocab_preview && data.vocab_preview.length > 0) {
                    vocabPreview.innerHTML = "";
                    data.vocab_preview.forEach(tok => {
                        const chip = document.createElement("div");
                        chip.className = "vocab-chip";
                        chip.textContent = tok;
                        vocabPreview.appendChild(chip);
                    });
                }
                
                // Show loss plot if exists
                if (data.has_loss_plot) {
                    const plotContainer = document.getElementById("plotContainer");
                    plotContainer.innerHTML = `<img src="/images/loss_plot.png?t=${Date.now()}" class="loss-plot-img" alt="Training Loss Curve">`;
                }
                
                // Enable audio playback if WAV is already generated
                if (data.has_wav) {
                    enablePlayback();
                }
            } catch (err) {
                console.error("Failed to load server status:", err);
            }
        }
        
        function enablePlayback() {
            btnPlay.disabled = false;
            dlWav.href = "/audio/generated_music.wav";
            dlWav.style.pointerEvents = "auto";
            dlWav.style.opacity = "1";
            
            dlMidi.href = "/audio/generated_music.mid";
            dlMidi.style.pointerEvents = "auto";
            dlMidi.style.opacity = "1";
        }
        
        // Compose music triggering API
        btnGenerate.addEventListener("click", async () => {
            loaderOverlay.classList.add("active");
            stopPlayback();
            btnPlay.disabled = true;
            
            try {
                const response = await fetch("/api/generate", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        gen_len: paramLen.value,
                        temperature: paramTemp.value,
                        bpm: paramBPM.value
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    // Force reloading WAV buffer
                    audioBuffer = null;
                    enablePlayback();
                    
                    // Show animation and toast success
                    setTimeout(() => {
                        loaderOverlay.classList.remove("active");
                    }, 500);
                } else {
                    alert("Generation failed: " + result.error);
                    loaderOverlay.classList.remove("active");
                }
            } catch (err) {
                alert("API communication error: " + err);
                loaderOverlay.classList.remove("active");
            }
        });
        
        // Play / Pause music engine
        btnPlay.addEventListener("click", () => {
            if (isPlaying) {
                pausePlayback();
            } else {
                playPlayback();
            }
        });
        
        async function loadAudioBuffer() {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
            
            if (audioBuffer) return; // already loaded
            
            // Show loading indicators inside play button
            btnPlay.textContent = "⏳";
            btnPlay.disabled = true;
            
            try {
                // Fetch the WAV file (bypass cache with timestamp)
                const res = await fetch(`/audio/generated_music.wav?t=${Date.now()}`);
                const arrayBuf = await res.arrayBuffer();
                audioBuffer = await audioContext.decodeAudioData(arrayBuf);
            } catch (e) {
                console.error("Failed to load/decode WAV audio file", e);
                alert("Audio decoding failed. Please regenerate the song.");
            } finally {
                btnPlay.textContent = "▶";
                btnPlay.disabled = false;
            }
        }
        
        async function playPlayback() {
            await loadAudioBuffer();
            if (!audioBuffer) return;
            
            if (audioContext.state === "suspended") {
                await audioContext.resume();
            }
            
            audioSource = audioContext.createBufferSource();
            audioSource.buffer = audioBuffer;
            
            // Create analyzer for visualization
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            
            audioSource.connect(analyser);
            analyser.connect(audioContext.destination);
            
            // Handle loop offset
            const offset = pausedAt;
            audioSource.start(0, offset);
            startTime = audioContext.currentTime - offset;
            isPlaying = true;
            btnPlay.textContent = "⏸";
            
            audioSource.onended = () => {
                // If it finished on its own
                if (audioContext.currentTime - startTime >= audioBuffer.duration) {
                    stopPlayback();
                }
            };
            
            startVisualizer();
        }
        
        function pausePlayback() {
            if (!isPlaying) return;
            audioSource.stop();
            pausedAt = audioContext.currentTime - startTime;
            isPlaying = false;
            btnPlay.textContent = "▶";
        }
        
        function stopPlayback() {
            if (isPlaying) {
                audioSource.stop();
            }
            pausedAt = 0;
            isPlaying = false;
            btnPlay.textContent = "▶";
        }
        
        // Canvas-based Waveform Visualizer
        const canvas = document.getElementById("visualizerCanvas");
        const ctx = canvas.getContext("2d");
        
        // Resize canvas to match container
        function resizeCanvas() {
            canvas.width = canvas.clientWidth;
            canvas.height = canvas.clientHeight;
        }
        window.addEventListener("resize", resizeCanvas);
        resizeCanvas();
        
        function startVisualizer() {
            if (!isPlaying || !analyser) return;
            
            const bufferLength = analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            
            function draw() {
                if (!isPlaying) {
                    // Reset to flat line
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.beginPath();
                    ctx.moveTo(0, canvas.height / 2);
                    ctx.lineTo(canvas.width, canvas.height / 2);
                    ctx.strokeStyle = "rgba(6, 182, 212, 0.4)";
                    ctx.lineWidth = 2;
                    ctx.stroke();
                    return;
                }
                
                requestAnimationFrame(draw);
                
                analyser.getByteTimeDomainData(dataArray);
                
                ctx.fillStyle = "#06060c";
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                ctx.lineWidth = 3;
                
                // Color gradient
                const grad = ctx.createLinearGradient(0, 0, canvas.width, 0);
                grad.addColorStop(0, "#8b5cf6");
                grad.addColorStop(0.5, "#ec4899");
                grad.addColorStop(1, "#06b6d4");
                ctx.strokeStyle = grad;
                
                ctx.beginPath();
                
                const sliceWidth = canvas.width / bufferLength;
                let x = 0;
                
                for (let i = 0; i < bufferLength; i++) {
                    const v = dataArray[i] / 128.0;
                    const y = v * canvas.height / 2;
                    
                    if (i === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                    
                    x += sliceWidth;
                }
                
                ctx.lineTo(canvas.width, canvas.height / 2);
                ctx.stroke();
            }
            
            draw();
        }
        
        // Reset static line
        startVisualizer();
        
        // Load initial status
        fetchStatus();
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Initializing Symphony AI Studio localhost server...")
    # Serve on port 5000 (standard development server)
    app.run(host="0.0.0.0", port=5000, debug=False)
