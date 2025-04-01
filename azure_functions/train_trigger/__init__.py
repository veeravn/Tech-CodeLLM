import logging
import azure.functions as func
from .continuous_train_trigger import trigger_training

def main(mytimer: func.TimerRequest) -> None:
    logging.info("[train_trigger] Starting training job...")
    trigger_training()