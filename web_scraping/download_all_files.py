import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import time
import mimetypes
import argparse


class WebScraper:
    def __init__(self, base_url, output_dir="downloaded_files", delay=1):
        """
        Initialize the web scraper with a base URL and output directory.

        Args:
            base_url (str): The base URL of the website to scrape
            output_dir (str): Directory where files will be saved
            delay (float): Delay between requests in seconds to avoid overwhelming the server
        """
        self.base_url = base_url
        self.output_dir = output_dir
        self.delay = delay
        self.domain = urlparse(base_url).netloc
        self.visited_urls = set()
        self.queue = [base_url]

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def is_valid_url(self, url):
        """Check if URL belongs to the same domain and hasn't been visited yet."""
        parsed_url = urlparse(url)
        return (parsed_url.netloc == self.domain or not parsed_url.netloc) and url not in self.visited_urls

    def is_text_file(self, url):
        """Check if the URL points to a text file (excluding HTML)."""
        # Get file extension from URL
        path = urlparse(url).path
        ext = os.path.splitext(path)[1].lower()

        # Common text file extensions
        text_extensions = ['.txt', '.md', '.csv', '.json', '.xml', '.log', '.ini', '.cfg',
                           '.yml', '.yaml', '.toml', '.rtf', '.conf', '.config', '.py',
                           '.js', '.css', '.c', '.cpp', '.h', '.java', '.rb', '.pl', '.php',
                           '.sh', '.bat', '.ps1', '.sql', '.r', '.tex', '.bib']

        # Excluded extensions (HTML files)
        excluded_extensions = ['.html', '.htm', '.xhtml', '.shtml']

        if ext in text_extensions and ext not in excluded_extensions:
            return True

        # If no extension, try to determine content type from headers
        if not ext:
            try:
                response = requests.head(url, allow_redirects=True, timeout=5)
                content_type = response.headers.get('Content-Type', '')

                # Check if it's a text file but not HTML
                return ('text/' in content_type.lower() and
                        'html' not in content_type.lower() and
                        'xhtml' not in content_type.lower())
            except Exception:
                return False

        return False

    def download_file(self, url):
        """Download a file from the given URL."""
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                # Extract filename from URL
                filename = os.path.basename(urlparse(url).path)

                # If filename is empty or doesn't have an extension, create one
                if not filename or '.' not in filename:
                    content_type = response.headers.get('Content-Type', '').split(';')[0]
                    extension = mimetypes.guess_extension(content_type) or '.txt'
                    filename = f"file_{len(self.visited_urls)}{extension}"

                # Save the file
                file_path = os.path.join(self.output_dir, filename)
                with open(file_path, 'wb') as f:
                    f.write(response.content)

                print(f"Downloaded: {filename} from {url}")
                return True
            else:
                print(f"Failed to download: {url} (Status code: {response.status_code})")
                return False
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
            return False

    def scrape(self, max_files=None):
        """
        Start the scraping process.

        Args:
            max_files (int, optional): Maximum number of files to download
        """
        files_downloaded = 0

        while self.queue and (max_files is None or files_downloaded < max_files):
            # Get the next URL from the queue
            current_url = self.queue.pop(0)

            # Skip if already visited
            if current_url in self.visited_urls:
                continue

            print(f"Processing: {current_url}")
            self.visited_urls.add(current_url)

            try:
                # Download if it's a text file
                if self.is_text_file(current_url):
                    if self.download_file(current_url):
                        files_downloaded += 1
                    time.sleep(self.delay)  # Be respectful to the server
                    continue  # Skip further processing for files

                # Get the webpage content
                response = requests.get(current_url, timeout=10)
                if response.status_code != 200:
                    continue

                # Parse HTML content
                soup = BeautifulSoup(response.text, 'html.parser')

                # Find all links
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    absolute_url = urljoin(current_url, href)

                    # Only add URLs from the same domain that haven't been visited
                    if self.is_valid_url(absolute_url):
                        self.queue.append(absolute_url)

                # Be respectful to the server
                time.sleep(self.delay)

            except Exception as e:
                print(f"Error processing {current_url}: {str(e)}")

        print(f"\nScraping complete. Downloaded {files_downloaded} files.")


def main():
    parser = argparse.ArgumentParser(description='Scrape and download text files from a website.')
    parser.add_argument('url', help='The base URL to start scraping from')
    parser.add_argument('--output', '-o', default='downloaded_files', help='Output directory for downloaded files')
    parser.add_argument('--delay', '-d', type=float, default=1.0, help='Delay between requests in seconds')
    parser.add_argument('--max', '-m', type=int, default=None, help='Maximum number of files to download')

    args = parser.parse_args()

    scraper = WebScraper(args.url, output_dir=args.output, delay=args.delay)
    scraper.scrape(max_files=args.max)


if __name__ == '__main__':
    main()