from src.api.routes import app
from verify_setup import verify_setup
import logging

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    if verify_setup():
        logger.info("Starting application...")
        app.run(debug=True)
    else:
        logger.error("Setup verification failed. Please check logs.")