import logging
import sys
from crypto_predictor.data.data_ingestion import ingest_data
from crypto_predictor.data.data_preprocessing import preprocess_data, prepare_sequences
from crypto_predictor.models.train import train_model
from crypto_predictor.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_pipeline(config_path: str) -> None:
    try:
        config = load_config(config_path)
        # Data ingestion
        raw_data = ingest_data(config.get("data", {}))
        # Data preprocessing
        processed_data = preprocess_data(raw_data, config.get("preprocessing", {}))
        # Prepare training sequences using sliding window approach
        window_size = config.get("preprocessing", {}).get("window_size", 60)
        prepared_data = prepare_sequences(processed_data, window_size)
        # Model training
        train_model(prepared_data, config.get("training", {}))
        logger.info("Pipeline completed successfully.")
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    config_path = "config/config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    run_pipeline(config_path)
