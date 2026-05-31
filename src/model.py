from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Bidirectional
from tensorflow.keras.optimizers import Adam

def build_lstm_model(vocab_size: int, seq_length: int, embedding_dim: int = 128, lstm_units: int = 256) -> Sequential:
    """
    Builds and compiles a deep learning LSTM model for musical note sequence prediction.
    Uses an Embedding layer followed by double Stacked LSTM layers with dropout regularization.
    """
    model = Sequential([
        # Embedding layer learns high-quality vector representations for musical tokens
        Embedding(
            input_dim=vocab_size, 
            output_dim=embedding_dim, 
            input_length=seq_length,
            name="note_embedding"
        ),
        
        # First LSTM Layer
        LSTM(
            lstm_units, 
            return_sequences=True, 
            recurrent_dropout=0.0,  # GPU compatible standard LSTM
            name="lstm_1"
        ),
        Dropout(0.3, name="dropout_1"),
        
        # Second LSTM Layer (return_sequences=False as it feeds into the final Dense layers)
        LSTM(
            lstm_units, 
            return_sequences=False,
            recurrent_dropout=0.0,
            name="lstm_2"
        ),
        Dropout(0.3, name="dropout_2"),
        
        # Intermediate dense layer for feature representations
        Dense(lstm_units // 2, activation="relu", name="dense_features"),
        Dropout(0.2, name="dropout_3"),
        
        # Output layer with softmax activation mapping to unique token probabilities
        Dense(vocab_size, activation="softmax", name="output_layer")
    ])
    
    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    
    print("Model successfully built and compiled.")
    model.summary()
    return model
