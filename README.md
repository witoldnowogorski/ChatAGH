### THIS REPOSITORY IS DEPRECATED, WORKS ARE CONTINUED HERE: https://github.com/ChatAGH


# About project

This project introduces an **advanced Agentic RAG (Retrieval-Augmented Generation)** architecture inspired by recent progress in agentic AI systems and surrounding technologies. It is designed to operate on web data under specific domain and leverages combination of **vector and graph database** to represent the structure of the data. The system **addresses one of the fundamental limitations** of traditional RAG systems - the lack of awareness of relationships present in the data and by doing so enables more context-aware and semantically rich reasoning over web data.

The system is being developed to operate on data sourced from websites affiliated with [AGH University of Science and Technology](https://www.agh.edu.pl), with the long-term goal of serving as a practical and intelligent tool for the university community.

Below is a brief overview of the system architecture, data scraping and processing pipelines, as well as the core technologies used. For a **developer guide** and more detailed descriptions please refer to the [documentation](https://github.com/witoldnowogorski/ChatAGH/tree/readme-update/docs).

## Data Collection, Processing, and Indexing
<img width="1000" height="600" alt="Screenshot 2025-07-16 at 20 01 13" src="https://github.com/user-attachments/assets/24bfff7e-79b3-4c0a-97db-251625252b87" />

The initialization of the system’s external knowledge base involves three main stages:

#### 1. Scraping
Given a specific domain, the system performs a full traversal of the URL graph, systematically collecting and filtering the content of all reachable web pages. This step includes:
- Crawling internal links within the domain.
- Extracting page content while applying filtering rules to eliminate irrelevant or duplicate data.
#### 2. Processing
In this stage, the system performs two key operations:
- Link Extraction – All links present in the content of each page are extracted. These may come from navigation menus, buttons, or inline references within the body of the page.
- Chunking and Embedding – Each page is segmented into smaller textual units ("chunks") of a predefined size, as in a standard RAG pipeline. Each chunk is enriched with metadata including:
the `source_url` from which it originated and `sequence_number` indicating its order within the original page.

#### 3. Indexing
In the final stage, the processed data is indexed into two separate storage systems:
- Graph Database – Stores relationships between pages in the form of edges between `source_url` and `target_url`, effectively capturing the hyperlink structure of the domain.
- Vector Database – Stores the content chunks along with their embeddings and associated metadata, enabling fast semantic search during the retrieval phase.


## System Architecture Overview

<img width="1000" height="550" alt="Screenshot 2025-07-16 at 20 01 05" src="https://github.com/user-attachments/assets/6d44ffbb-eed9-4c26-91f8-df611dd98fd6" />

The diagram above presents the core component of the system responsible for retrieval, reasoning, and answer generation. This architecture integrates multiple specialized modules to enable context-aware, accurate, and source-grounded responses.

#### Input
 - The conversation history between the user and the system.
 - The cached retrieved context, containing information previously extracted from the external knowledge base during earlier user's interactions with the system.

#### RAG Router
In the first step of the reasoning process, the RAG Router evaluates whether the current context — built from the conversation history and cached knowledge — is sufficient to answer the user's message. If the current context is sufficient to generate a response, the system proceeds without additional retrieval. Otherwise, it activates the retrieval pathway to query the external knowledge base and extract missing information.

#### Query Generator
If the retrieval pathway was chosen the comprehensive query to the external knowledge base is generated. The LLM analyzes the query and cached context to determine missing data and provide most suitable query.

#### Similarity Search
Generated query is used to perform similarity search against chunks in the vector store, hybrid search techniques are leveraged to find the most accurate, relevant chunks.
The retrieved chunks are grouped by their corresponding source_url. The result of this step can be represented as a dictionary-like structure, where each key is a URL (representing a node in the web graph), and the associated value is a list of retrieved chunks originating from that url.

#### Context Analyzer
In the next step, the system filters out groups of chunks to retain only those that are relevant to the user's question. For each group classified as relevant, a summary is generated, capturing all key information that may contribute to answering the query. Again, the result of this step can be represented as a dictionary-like structure, where each key is a URL, and the associated value is summary of the relevant url content.

#### Context Augmentation
This step is critical for enabling the system to reason over a broader context and uncover deeper relationships between data.

For each group retained after summarization, the system queries the graph database to identify all pages directly linked to the group's source_url. The summary of the original group is then embedded and compared against all chunks associated with these neighboring nodes in the vector database.

The chuks retrieved this way can be passed back into the Context Analyzer, effectively creating a loop of retrieval and analysis that can traverse the web graph iteratively. This mechanism allows the system to explore and incorporate increasingly distant but contextually relevant pages, enriching the information available for final answer generation.

#### Answer Generation
Based on the conversation history and the retrieved context, the system generates a response to the user’s latest message. The reply can range from a simple greeting to a detailed answer with source references, depending on the user’s intent and the available information.

## Technologies 
The system is built using modern tools that enable scalable retrieval and seamless integration with large language models:

- MongoDB – Serves both as a vector store (via Atlas Vector Search) and a graph data store, enabling efficient semantic search and relationship retrieval across web content.
- Gemini API –  A powerful interface to Google’s multimodal large language models, capable of processing text and other data types.
LangGraph / LangChain – Frameworks used to define and orchestrate the system’s modular reasoning pipeline. They allow the implementation of conditional routing, memory management, and multi-step workflows aligned with the Agentic RAG architecture.

