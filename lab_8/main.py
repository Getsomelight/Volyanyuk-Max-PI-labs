import logging
from dotenv import load_dotenv
from image_downloader import *


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Handle argument and execute image download
    """
    logger.info("Loading environment")
    load_dotenv()
    logger.info(f"Starting async download from: {os.getenv('CSV_URLS')}")
    asyncio.run(download_images_from_csv(logger, os.getenv('CSV_URLS'), os.getenv('FOLDER'), int(os.getenv('MAX_CONCURRENT'))))


if __name__ == "__main__":
    main()