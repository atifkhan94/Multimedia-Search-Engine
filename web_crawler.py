from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import requests
import logging
from typing import List, Dict, Optional

class WebCrawler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()

    def setup_driver(self) -> webdriver.Chrome:
        """Initialize and return a Chrome WebDriver instance."""
        try:
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            return webdriver.Chrome(options=options)
        except Exception as e:
            self.logger.error(f"Failed to initialize WebDriver: {e}")
            raise

    def crawl_page(self, url: str) -> Dict[str, any]:
        """Crawl a single webpage and extract multimedia content.

        Args:
            url (str): The URL to crawl

        Returns:
            Dict[str, any]: Dictionary containing extracted content
        """
        try:
            driver = self.setup_driver()
            driver.get(url)

            # Extract images
            images = self._extract_images(driver)
            
            # Extract videos
            videos = self._extract_videos(driver)
            
            # Extract text content
            text_content = self._extract_text(driver)

            return {
                'url': url,
                'images': images,
                'videos': videos,
                'text': text_content
            }

        except Exception as e:
            self.logger.error(f"Error crawling {url}: {e}")
            return {}
        finally:
            driver.quit()

    def _extract_images(self, driver: webdriver.Chrome) -> List[Dict[str, str]]:
        """Extract image URLs and alt text from the page."""
        images = []
        for img in driver.find_elements(By.TAG_NAME, 'img'):
            try:
                images.append({
                    'url': img.get_attribute('src'),
                    'alt': img.get_attribute('alt') or ''
                })
            except:
                continue
        return images

    def _extract_videos(self, driver: webdriver.Chrome) -> List[Dict[str, str]]:
        """Extract video URLs from the page."""
        videos = []
        video_elements = driver.find_elements(By.TAG_NAME, 'video')
        video_elements.extend(driver.find_elements(By.TAG_NAME, 'iframe'))

        for video in video_elements:
            try:
                src = video.get_attribute('src')
                if src:
                    videos.append({
                        'url': src,
                        'type': 'video'
                    })
            except:
                continue
        return videos

    def _extract_text(self, driver: webdriver.Chrome) -> str:
        """Extract main text content from the page."""
        try:
            return driver.find_element(By.TAG_NAME, 'body').text
        except:
            return ""