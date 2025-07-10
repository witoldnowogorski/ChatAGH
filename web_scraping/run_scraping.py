from scraper import Scraper
from urls_graph_generator import GraphGenerator


if __name__ == "__main__":
    crawler = GraphGenerator(
        allowed_domains=["agh.edu.pl"],
        start_urls=[
            "https://www.agh.edu.pl",
            "https://rekrutacja.agh.edu.pl",
            "https://www.eaiib.agh.edu.pl",
            "https://www.sjo.agh.edu.pl",
            "https://www.swfis.agh.edu.pl",
            "https://sylabusy.agh.edu.pl",
            "https://sylabusy.agh.edu.pl/pl/",
            "https://skn.agh.edu.pl",
            "https://dss.agh.edu.pl",
            "https://akademik.agh.edu.pl",
            "https://www.miasteczko.agh.edu.pl"
        ],
        max_pages=100
    )
    crawler.generate_graph()
    crawler.graph_to_json()

    urls = crawler.get_nodes()

    scraper = Scraper(urls=urls)
    scraper.scrape()
    scraper.save_result_to_json("documents.json")
    scraper.save_docs_as_md()
