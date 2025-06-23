import re
import json
import copy
import tempfile
import requests
import datetime
from pathlib import Path

import html2text
import pytesseract
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from pdf2image import convert_from_path
from langchain_core.documents import Document


class Scraper:
    def __init__(self, urls, output_path="downloads/"):
        self.urls = urls
        self.output_path = output_path

        self.processed_urls = []
        self.failed_urls = []
        self.documents = []

    def scrape(self):
        for i, url in enumerate(self.urls):
            print("Processing url {}/{}".format(i+1, len(self.urls)))
            try:
                if "doc.php" in url:
                    self.download_php_doc(url)
                else:
                    try:
                        self.download_pdf(url)
                    except Exception as e:
                        self.download_html_content(url)

                self.processed_urls.append(url)

            except Exception as e:
                print(f"Failed to process url: {e}")
                self.failed_urls.append(url)

    def save_result_to_json(self, filename):
        data = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in self.documents
        ]
        with open(Path(self.output_path) / filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save_docs_as_md(self):
        for doc in self.documents:
            url = doc.metadata["url"]
            filename = url.replace("/", "_") + ".md"
            with open(Path(self.output_path) / filename, "w") as f:
                f.write(doc.page_content)

    def download_pdf(self, url):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_pdf_path = f"{tmpdir}/temp.pdf"
                with open(tmp_pdf_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                pages = convert_from_path(tmp_pdf_path)

                markdown = ""
                for page in pages:
                    text = pytesseract.image_to_string(page)
                    markdown += f"\n{text}\n"

            document = Document(page_content=markdown, metadata={"url": url})
            self.documents.append(document)

    def download_php_doc(self, url):
        response = requests.get(url, stream=True, allow_redirects=True, timeout=10)
        content = str(response.content)
        content.find(".pdf")
        file_path = content[:content.find(".pdf")].split('"')[-1] + ".pdf"
        processed_url = urlparse(url).scheme + "://" + urlparse(url).netloc + "/" + file_path
        response = requests.get(processed_url, stream=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_pdf_path = f"{tmpdir}/temp.pdf"
            with open(tmp_pdf_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            pages = convert_from_path(tmp_pdf_path)

            markdown = ""
            for page in pages:
                text = pytesseract.image_to_string(page)
                markdown += f"\n{text}\n"

        document = Document(page_content=markdown, metadata={"url": processed_url})
        self.documents.append(document)

    def download_html_content(self, url):
        """
        Download HTML content from a URL and extract the relevant content.
        Improved for better content detection and deduplication.
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return f"Error fetching URL: {str(e)}"

        try:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Clean the HTML
            self._clean_html(soup)

            # Extract title
            title = self._extract_title(soup)

            # Extract and score content blocks
            content_blocks = self._extract_content_blocks(soup)

            # If no content found, try fallback methods
            if not content_blocks:
                content_blocks = self._extract_fallback_content(soup)

            if not content_blocks:
                return "No content found on the page."

            # Convert to markdown with deduplication
            markdown_content = self._html_to_markdown(title, content_blocks, url)

            document = Document(page_content=markdown_content.strip(), metadata={"url": url})
            self.documents.append(document)

            return document
        except Exception as e:
            return f"Error processing HTML content: {str(e)}"

    def _clean_html(self, soup):
        """Remove unnecessary elements from HTML."""
        # Remove script, style tags and comments
        for element in soup(['script', 'style', 'noscript', 'svg', 'iframe', 'head', 'meta']):
            element.decompose()

        # Remove hidden elements
        for element in soup.find_all(style=re.compile('display:\s*none|visibility:\s*hidden')):
            element.decompose()

        # Remove elements with specific class or id patterns
        noise_patterns = [
            'cookie', 'popup', 'banner', 'ad-', '-ad', 'advertisement',
            'notification', 'subscribe', 'newsletter', 'promo', 'share',
            'related-', 'comment', 'footer', 'header', 'nav', 'menu',
            'sidebar', 'widget', 'toolbar', 'modal'
        ]

        for pattern in noise_patterns:
            for element in soup.find_all(class_=re.compile(pattern, re.I)):
                element.decompose()
            for element in soup.find_all(id=re.compile(pattern, re.I)):
                element.decompose()

    def _extract_title(self, soup):
        """Extract page title."""
        title = ""

        # Try to get title from specific meta tags first
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            title = og_title['content'].strip()
        else:
            # Try to get the title from the title tag
            title_elem = soup.find('title')
            if title_elem:
                title = title_elem.get_text().strip()
            else:
                # Try to find the main heading
                h1 = soup.find('h1')
                if h1:
                    title = h1.get_text().strip()

        if title:
            return f"# {title}\n\n"
        return ""

    def _extract_content_blocks(self, soup):
        """
        Extract content blocks from the page with improved relevance detection.
        Returns a list of elements scored by relevance.
        """
        scored_blocks = []

        # First try schema.org structured data
        article_body = soup.find('div', itemprop='articleBody')
        if article_body:
            return [(100, article_body)]

        # Look for elements with specific semantic HTML5 tags
        priority_containers = soup.find_all(['main', 'article', 'section'])
        for container in priority_containers:
            if self._is_content_container(container):
                # Calculate content density score
                text_length = len(container.get_text(strip=True))
                tag_count = len(container.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']))
                if tag_count == 0:
                    continue

                content_density = text_length / tag_count
                tag_ratio = tag_count / max(1, len(container.find_all()))

                # Score based on content signals
                links = container.find_all('a')
                link_ratio = len(links) / max(1, tag_count)
                link_text_ratio = sum(len(a.get_text(strip=True)) for a in links) / max(1, text_length)

                # Calculate final score
                score = (content_density * 0.4 +
                         tag_ratio * 30 +
                         (1 - link_ratio) * 15 +
                         (1 - link_text_ratio) * 15)

                # Bonus for article tag
                if container.name == 'article':
                    score += 20

                scored_blocks.append((score, container))

        # If we have good candidates, return them
        if scored_blocks:
            # Return the top 1-2 blocks, depending on score differential
            scored_blocks.sort(reverse=True, key=lambda x: x[0])
            if len(scored_blocks) > 1 and scored_blocks[0][0] > scored_blocks[1][0] * 1.5:
                return [scored_blocks[0]]  # Return only the top block if it's significantly better
            elif len(scored_blocks) > 1:
                return scored_blocks[:2]  # Return top two blocks
            return [scored_blocks[0]]

        return []

    def _extract_fallback_content(self, soup):
        """Fallback methods for content extraction when primary methods fail."""
        scored_blocks = []

        # Try content divs by class/id name
        content_divs = []
        for div in soup.find_all('div'):
            if not div.attrs:
                continue

            attrs = []
            if div.has_attr('class'):
                attrs.extend(div['class'])
            if div.has_attr('id'):
                attrs.append(div['id'])

            attrs = [attr.lower() for attr in attrs if attr]

            content_indicators = ['content', 'article', 'post', 'entry', 'body', 'text', 'main']
            if any(indicator in attr for attr in attrs for indicator in content_indicators):
                # Score by text density and paragraph count
                paragraphs = div.find_all('p')
                if not paragraphs:
                    continue

                text_length = len(div.get_text(strip=True))
                p_count = len(paragraphs)
                avg_p_length = text_length / max(1, p_count)

                if p_count >= 3 and avg_p_length > 30:
                    score = avg_p_length * 0.2 + p_count * 2
                    scored_blocks.append((score, div))

        if scored_blocks:
            scored_blocks.sort(reverse=True, key=lambda x: x[0])
            return [scored_blocks[0]]

        # Last resort: find all paragraphs with substantial content
        paragraphs = soup.find_all('p')
        if paragraphs:
            content_paragraphs = [p for p in paragraphs if len(p.get_text(strip=True)) > 80]
            if content_paragraphs:
                # Group paragraphs that are likely part of the same content block
                grouped_paragraphs = self._group_paragraphs(content_paragraphs)
                if grouped_paragraphs:
                    return [(50, group) for group in grouped_paragraphs if len(group.get_text(strip=True)) > 200]

        return []

    def _group_paragraphs(self, paragraphs):
        """Group paragraphs that are likely part of the same content block."""
        if not paragraphs:
            return []

        # Try to find a common parent that contains most paragraphs
        parents = {}
        for p in paragraphs:
            for parent in p.parents:
                if parent.name in ['div', 'section', 'article', 'main']:
                    if parent not in parents:
                        parents[parent] = 0
                    parents[parent] += 1

        if not parents:
            return []

        # Find parent with most paragraphs
        best_parent = max(parents.items(), key=lambda x: x[1])[0]
        if parents[best_parent] >= len(paragraphs) * 0.7:  # If it contains at least 70% of paragraphs
            return [best_parent]

        # Create artificial div with all paragraphs if no good parent found
        container = BeautifulSoup('<div></div>', 'html.parser').div
        for p in paragraphs:
            container.append(copy.copy(p))
        return [container]

    def _is_content_container(self, element):
        """Determine if an element is likely to be a content container."""
        # Skip empty elements
        if not element.get_text(strip=True):
            return False

        # Skip small text blocks
        if len(element.get_text(strip=True)) < 200:
            return False

        # Skip containers with few block-level elements
        block_elements = element.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'blockquote'])
        if len(block_elements) < 3:
            return False

        # Check for negative patterns in attributes
        skip_patterns = [
            'header', 'footer', 'nav', 'menu', 'sidebar', 'banner',
            'advertisement', 'cookie', 'popup', 'modal', 'social',
            'comment', 'widget', 'toolbar', 'masthead'
        ]

        for attr in element.attrs:
            if attr in ['class', 'id'] and element.has_attr(attr):
                values = element[attr] if isinstance(element[attr], list) else [element[attr]]
                for value in values:
                    if any(pattern in value.lower() for pattern in skip_patterns):
                        return False

        # Check for navigation roles
        if element.has_attr('role'):
            role = element['role'].lower()
            skip_roles = ['navigation', 'banner', 'complementary', 'contentinfo']
            if role in skip_roles:
                return False

        return True

    def _html_to_markdown(self, title, content_blocks, url):
        """Convert HTML content to markdown with deduplication."""
        # Initialize HTML2Text converter
        converter = html2text.HTML2Text()
        converter.ignore_links = False
        converter.ignore_images = False
        converter.ignore_tables = False
        converter.body_width = 0

        markdown_content = title

        # Keep track of processed text to avoid duplication
        seen_content = set()

        for _, element in content_blocks:
            # Process each element and avoid duplicating content
            element_html = str(element)
            md_part = converter.handle(element_html)

            # Clean up markdown
            md_part = re.sub(r'\n{3,}', '\n\n', md_part)

            # Split into paragraphs and process each to avoid duplication
            paragraphs = re.split(r'\n\n+', md_part)
            unique_paragraphs = []

            for paragraph in paragraphs:
                # Normalize paragraph for deduplication check
                normalized = self._normalize_text(paragraph)
                if normalized and normalized not in seen_content and len(normalized) > 20:
                    unique_paragraphs.append(paragraph)
                    seen_content.add(normalized)

            # Join unique paragraphs back together
            if unique_paragraphs:
                markdown_content += '\n\n'.join(unique_paragraphs) + '\n\n'

        # Add source URL at the end
        markdown_content += f"\n\n---\nSource: {url}"

        return markdown_content.strip()

    def _normalize_text(self, text):
        """Normalize text for deduplication comparison."""
        if not text or not text.strip():
            return ""

        # Remove extra whitespace, markdown formatting, and convert to lowercase
        normalized = re.sub(r'\s+', ' ', text)
        normalized = re.sub(r'[#*_\[\]()~`>|-]', '', normalized)
        normalized = normalized.lower().strip()

        # For very short strings, return as is
        if len(normalized) < 20:
            return normalized

        # For longer strings, use the first 100 chars to compare
        return normalized[:100]

    def _get_output_filename(self, url):
        """Generate a clean filename from the URL."""
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        path = parsed.path.strip("/").replace("/", "_")
        if not path:
            path = "index"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"{domain}_{path}_{timestamp}.md"
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)

        if len(filename) > 240:
            filename = filename[:240]

        return filename


if __name__ == "__main__":
    scraper = Scraper(urls=
        [
            "http://www.agh.edu.pl/wydarzenia/detail/s/to-bedzie-maj-juwekrk"
         ])

    scraper.scrape()
    scraper.save_result_to_json("documents.json")
    scraper.save_docs_as_md()