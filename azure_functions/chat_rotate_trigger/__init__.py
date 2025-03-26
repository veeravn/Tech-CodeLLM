import logging
import azure.functions as func
from .rotate_chat_history import rotate_chat_history

def main(mytimer: func.TimerRequest) -> None:
    logging.info("[chat_rotate_trigger] Rotating chat history...")
    rotate_chat_history()
