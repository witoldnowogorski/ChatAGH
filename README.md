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


