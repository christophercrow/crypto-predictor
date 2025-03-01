import logging
from tensorflow.keras.models import load_model
from crypto_predictor.data.data_ingestion import ingest_data
from crypto_predictor.data.data_preprocessing import preprocess_data, prepare_sequences
from crypto_predictor.utils.config import load_config

# Configure logging to ensure messages are visible
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(config_path: str = "config/config.yaml") -> None:
    """
    Load test data, prepare sequences, load the saved model, and evaluate its performance.
    
    Args:
        config_path: Path to the configuration YAML file.
    """
    logger.info("Starting model evaluation")
    config = load_config(config_path)
    
    # Ingest and preprocess data (for evaluation, you might want a separate test set)
    raw_data = ingest_data(config.get("data", {}))
    processed_data = preprocess_data(raw_data, config.get("preprocessing", {}))
    
    # Prepare sequences
    window_size = config.get("preprocessing", {}).get("window_size", 60)
    sequences = prepare_sequences(processed_data, window_size)
    X, y = sequences["X_train"], sequences["y_train"]
    
    # Load the trained model
    save_path = config.get("training", {}).get("save_path", "models/saved_model.keras")
    if not (save_path.endswith(".keras") or save_path.endswith(".h5")):
        save_path = save_path + ".keras"
    model = load_model(save_path)
    
    # Evaluate the model on the test sequences
    loss = model.evaluate(X, y, verbose=0)
    logger.info("Evaluation Loss: %f", loss)
    print(f"Evaluation Loss: {loss}")

if __name__ == "__main__":
    evaluate_model()
