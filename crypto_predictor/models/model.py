import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from typing import Dict, Any

def build_model(training_config: Dict[str, Any]) -> tf.keras.Model:
    """
    Build and compile an LSTM model based on the training configuration.
    Args:
        training_config: Dictionary with model parameters.
    Returns:
        A compiled TensorFlow Keras model.
    """
    model = Sequential()
    input_shape = training_config.get("input_shape", (None, 1))
    lstm_units = training_config.get("lstm_units", 50)

    model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=lstm_units))
    model.add(Dense(1))
    
    model.compile(
        optimizer=training_config.get("optimizer", "adam"),
        loss=training_config.get("loss", "mse")
    )
    return model
