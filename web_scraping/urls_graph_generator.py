import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import networkx as nx
from urllib.parse import urlparse
import plotly.graph_objects as go
from tqdm import tqdm
import time
import random


class GraphGenerator:
    def __init__(self, start_urls, allowed_domains, max_pages=10):
        self.allowed_domains = set(allowed_domains)
        self.start_urls = set(start_urls)
        self.max_pages = max_pages
        self.visited_urls = set()
        self.G = nx.DiGraph()

    def generate_graph(self):
        for start_url in self.start_urls:
            print(start_url)
            self.crawl(start_url, self.max_pages)
        return self.G

    def crawl(self, start_url, max_pages, delay=1):
        """
        Crawl the web starting from start_url, only following links within allowed_domains.

        Parameters:
        - start_url: The URL to start crawling from
        - allowed_domains: List of allowed domains to crawl
        - max_pages: Maximum number of pages to crawl
        - delay: Delay between requests in seconds

        Returns:
        - G: NetworkX graph of the crawled web
        - visited_urls: Set of visited URLs
        """
        start_domain = urlparse(start_url).netloc
        self.allowed_domains.add(start_domain)

        queue = [start_url]

        pbar = tqdm(total=max_pages, desc="Crawling")

        visited = []

        while queue and len(visited) < max_pages:
            current_url = queue.pop(0)

            if current_url in visited or current_url in self.visited_urls:
                continue

            extensions_to_filter = [".jpg"]
            for extensions in extensions_to_filter:
                if current_url.endswith(extensions):
                    continue

            visited.append(current_url)
            pbar.update(1)

            try:
                links_df = self.extract_links_from_url(current_url)

                if links_df is not None and not links_df.empty:
                    for link in links_df['url']:
                        if self.is_allowed_domain(link):
                            self.G.add_edge(current_url, link)
                            if link not in visited and link not in self.visited_urls:
                                queue.append(link)

                if current_url not in self.G:
                    self.G.add_node(current_url)

                time.sleep(delay + random.uniform(0, 0.5))

            except Exception as e:
                print(f"Error crawling {current_url}: {e}")

        for url in visited:
            self.visited_urls.add(url)

        pbar.close()


    def is_allowed_domain(self, url):
        """Check if the URL belongs to one of the allowed domains."""
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            return domain in self.allowed_domains or any(domain.endswith('.' + d) for d in self.allowed_domains)
        except:
            return False

    def extract_links_from_url(self, url):
        try:
            headers = {
                'User-Agent':
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching the URL: {e}")
            return pd.DataFrame(columns=['link_text', 'url', 'link_type', 'file_extension'])

        soup = BeautifulSoup(response.text, 'html.parser')

        base_url = url

        link_urls = []
        file_extensions = []

        for a_tag in soup.find_all('a', href=True):
            href = a_tag.get('href')
            if not href or href.startswith(('javascript:', '#')):
                continue

            absolute_url = urljoin(base_url, href)

            parsed_url = urlparse(absolute_url)
            path = parsed_url.path.lower()
            file_extension = path.split('.')[-1] if '.' in path else ''

            link_urls.append(absolute_url)
            file_extensions.append(file_extension if file_extension else "None")

        df = pd.DataFrame({
            'url': link_urls,
            'file_extension': file_extensions
        })

        df = df.drop_duplicates(subset=['url'])

        return df

    def analyze_graph(self):
        """Analyze the graph and return some statistics."""
        stats = {
            "Number of nodes": len(self.G.nodes()),
            "Number of edges": len(self.G.edges()),
            "Average out-degree": sum(dict(self.G.out_degree()).values()) / len(self.G.nodes()) if self.G.nodes() else 0,
            "Average in-degree": sum(dict(self.G.in_degree()).values()) / len(self.G.nodes()) if self.G.nodes() else 0
        }

        if len(self.G.nodes()) > 0:
            in_degrees = dict(self.G.in_degree())
            top_inbound = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
            stats["Top pages by incoming links"] = top_inbound

            out_degrees = dict(self.G.out_degree())
            top_outbound = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
            stats["Top pages by outgoing links"] = top_outbound

            try:
                if nx.is_strongly_connected(self.G):
                    stats["Diameter"] = nx.diameter(self.G)
                else:
                    largest_cc = max(nx.strongly_connected_components(self.G), key=len)
                    subgraph = self.G.subgraph(largest_cc)
                    stats["Diameter (largest component)"] = nx.diameter(subgraph)
            except:
                stats["Diameter"] = "Could not compute (graph may not be connected)"

        return stats

    def get_visualization(self):
        pos = nx.spring_layout(self.G, seed=42)

        edge_x = []
        edge_y = []
        for u, v in self.G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo='none',
            mode='lines'
        )

        node_x = []
        node_y = []
        labels = []
        for node in self.G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            labels.append(str(node))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=labels,
            marker=dict(
                showscale=False,
                color="#1f77b4",
                size=10,
                line_width=2
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Interactive Graph Visualization',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))

        fig.show()

    def graph_to_json(self):
        nodes = [{"url": node} for node in self.G.nodes]
        edges = [{"source": u, "target": v} for u, v in self.G.edges]

        with open("graph.json", "w") as f:
            json.dump({
                "nodes": nodes,
                "edges": edges
            }, f, indent=4)

    def get_nodes(self):
        return list(self.G.nodes())
