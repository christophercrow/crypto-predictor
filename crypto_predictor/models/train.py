import logging
from typing import Dict, Any
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from crypto_predictor.models.model import build_model

logger = logging.getLogger(__name__)

def train_model(prepared_data: Dict[str, Any], training_config: Dict[str, Any]) -> None:
    """
    Train the model using prepared data and training configuration.
    
    Args:
        prepared_data: Dictionary with 'X_train' and 'y_train'.
        training_config: Dictionary with training parameters.
    """
    try:
        logger.info("Starting model training")
        model = build_model(training_config)
        
        # Define callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        save_path = training_config.get("save_path", "models/saved_model")
        if not (save_path.endswith(".keras") or save_path.endswith(".h5")):
            logger.warning("Save path '%s' does not have a valid extension. Appending '.keras'", save_path)
            save_path = save_path + ".keras"
        checkpoint = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)
        
        history = model.fit(
            prepared_data["X_train"],
            prepared_data["y_train"],
            epochs=training_config.get("epochs", 10),
            batch_size=training_config.get("batch_size", 32),
            validation_split=training_config.get("validation_split", 0.2),
            callbacks=[early_stop, checkpoint]
        )
        logger.info("Model training completed")
    except Exception as e:
        logger.error("Model training error: %s", e)
        raise
