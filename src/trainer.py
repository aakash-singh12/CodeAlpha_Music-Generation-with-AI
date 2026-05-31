import os
from typing import Dict, Any
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train_model(
    model: Sequential, 
    X, 
    y, 
    epochs: int = 40, 
    batch_size: int = 64, 
    save_dir: str = "models"
) -> Dict[str, Any]:
    """
    Trains the sequential Keras model with ModelCheckpoint and EarlyStopping callbacks.
    Saves the best weights/model to the specified directory.
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "music_lstm_model.keras")
    
    # Callback to save the model with the minimum loss
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor="loss",
        verbose=1,
        save_best_only=True,
        mode="min"
    )
    
    # Callback to stop training early if training loss does not improve for 5 epochs
    early_stop = EarlyStopping(
        monitor="loss",
        patience=8,
        verbose=1,
        mode="min"
    )
    
    print(f"Starting model training for {epochs} epochs with batch size {batch_size}...")
    history = model.fit(
        X, 
        y, 
        epochs=epochs, 
        batch_size=batch_size, 
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    
    print(f"Training completed. Best model saved to: {checkpoint_path}")
    return history.history

def plot_training_loss(history_dict: Dict[str, Any], save_path: str = "output/loss_plot.png"):
    """
    Plots the training loss curve using Matplotlib and saves the figure.
    """
    if "loss" not in history_dict:
        print("Error: No loss metrics found in training history.")
        return
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(history_dict["loss"], label="Training Loss", color="#4f46e5", linewidth=2.5)
    
    if "accuracy" in history_dict:
        plt.plot(history_dict["accuracy"], label="Training Accuracy", color="#10b981", linestyle="--", linewidth=1.5)
        
    plt.title("AI Music Generator - Model Training Progress", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training history plot successfully saved to {save_path}")
