import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import pandas as pd

def extract_links(url):
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return pd.DataFrame(columns=['link_text', 'url', 'link_type', 'file_extension'])

    soup = BeautifulSoup(response.text, 'html.parser')

    base_url = url

    link_texts = []
    link_urls = []
    link_types = []
    file_extensions = []

    for a_tag in soup.find_all('a', href=True):
        href = a_tag.get('href')
        if not href or href.startswith(('javascript:', '#')):
            continue

        absolute_url = urljoin(base_url, href)
        link_text = a_tag.get_text(strip=True) or "[No Text]"

        parsed_url = urlparse(absolute_url)
        path = parsed_url.path.lower()
        file_extension = path.split('.')[-1] if '.' in path else ''

        if file_extension in ['pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'txt', 'csv']:
            link_type = "Download"
        else:
            link_type = "Hyperlink"

        link_texts.append(link_text)
        link_urls.append(absolute_url)
        link_types.append(link_type)
        file_extensions.append(file_extension if file_extension else "None")

    df = pd.DataFrame({
        'link_text': link_texts,
        'url': link_urls,
        'link_type': link_types,
        'file_extension': file_extensions
    })

    df = df.drop_duplicates(subset=['url'])

    return df

if __name__ == "__main__":
    links = extract_links("https://sylabusy.agh.edu.pl/pl/1/2/21/1/4/3/8")
    for link in links["url"]:
        print(link)