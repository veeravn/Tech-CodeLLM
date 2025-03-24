import logging
import azure.functions as func
from .scrape import scrape_and_trigger

def main(mytimer: func.TimerRequest) -> None:
    logging.info("⏰ Scraping function triggered by timer.")

    if mytimer.past_due:
        logging.warning("⚠️ Timer is running late!")

    try:
        scrape_and_trigger()
        logging.info("✅ Scraping and training job completed.")
    except Exception as e:
        logging.error(f"❌ Error during scraping or training: {str(e)}")
