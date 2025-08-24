from scraper import Scraper
from urls_graph_generator import GraphGenerator


if __name__ == "__main__":
    crawler = GraphGenerator(
        allowed_domains=["https://www.agh.edu.pl/"],
        start_urls=[
            "https://www.agh.edu.pl/"
        ],
        max_pages=50
    )
    crawler.generate_graph()
    crawler.graph_to_json()

    urls = crawler.get_nodes()

    scraper = Scraper(urls=urls)
    scraper.scrape()
    scraper.save_result_to_json("documents.json")
    scraper.save_docs_as_md()
