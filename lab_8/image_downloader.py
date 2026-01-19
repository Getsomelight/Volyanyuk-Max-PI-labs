import os
import csv
import asyncio
import aiohttp
from icrawler.builtin import GoogleImageCrawler


async def download_image(logger, session: aiohttp.ClientSession, url: str, filepath: str, timeout: int = 10) -> bool:
    """
    Downloads a single image asynchronously

    Args:
        logger : Logging info
        session: aiohttp ClientSession
        url (str): URL of the image to download
        filepath (str): Path where the image will be saved
        timeout (int): Download timeout in seconds

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            if response.status == 200:
                with open(filepath, 'wb') as f:
                    f.write(await response.read())
                logger.info(f"Downloaded: {os.path.basename(filepath)}")
                return True
            else:
                logger.info(f"Failed (status {response.status}): {url}")
                return False
    except asyncio.TimeoutError:
        logger.error(f"Timeout: {url}")
        return False
    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        return False


async def download_images_from_csv(logger, csv_path: str, folder: str, max_concurrent: int = 5) -> None:
    """
    Reads URLs from CSV file and downloads images asynchronously

    Args:
        csv_path (str): Path to CSV file with URLs in first column
        folder (str): Directory where images will be saved
        max_concurrent (int): Maximum concurrent downloads
        logger : Logging info
    """
    if not os.path.isdir(folder):
        try:
            os.makedirs(folder)
        except Exception:
            logger.error(f"Error creating directory")

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            urls = [row[0] for row in reader if row and row[0].strip()]
        logger.info("Csv read successfully")
    except FileNotFoundError:
        logger.error(f"Error: CSV file '{csv_path}' not found")
        return

    if not urls:
        logger.error("No URLs found in CSV file")
        return

    logger.info(f"Found {len(urls)} URLs to download")

    connector = aiohttp.TCPConnector(limit_per_host=3)
    async with aiohttp.ClientSession(connector=connector) as session:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def download_with_semaphore(url: str, index: int) -> bool:
            async with semaphore:
                try:
                    filename = os.path.basename(url.split('?')[0])
                    if not filename or '.' not in filename:
                        filename = f"image_{index}.jpg"
                except:
                    filename = f"image_{index}.jpg"

                filepath = os.path.join(folder, filename)

                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(filepath):
                    filepath = os.path.join(folder, f"{base}_{counter}{ext}")
                    counter += 1

                return await download_image(logger, session, url, filepath)

        tasks = [download_with_semaphore(url, i) for i, url in enumerate(urls)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = sum(1 for r in results if r is True)
    logger.info(f"Download complete: {successful}/{len(urls)} images downloaded successfully")