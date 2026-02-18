# STREAMLIT LINK: https://rdf-navigator-atgyrgyiifxrfyedxxnvkc.streamlit.app/ #
# 🕸️ Unified RDF Navigator: AI-Driven Knowledge Graph Explorer

An advanced Semantic Web application designed to transform heterogeneous tabular data into a structured **Knowledge Graph (RDF)** and enable natural language exploration via **Generative AI**.

---

## 🚀 Project Overview
This project addresses the challenge of making complex relational data more interpretable and queryable. By converting standard datasets (CSV, Excel, JSON) into **RDF triples**, the application creates a network of interconnected entities. Integration with the **Google Gemini Pro** model allows users to bypass complex SPARQL syntax and query the database using plain English.

## ✨ Core Features
* **Dynamic Triple Generation**: Automatically maps tabular columns to RDF predicates and rows to unique resource identifiers.
* **Natural Language to SPARQL**: Real-time translation of user questions into valid SPARQL queries using LLM-based RAG (Retrieval-Augmented Generation).
* **Interactive Visual Analytics**: Dynamic, color-coded graph visualization using PyVis for structural exploration of data relationships.
* **Semantic Reasoning Engine**: Built-in support for RDFS reasoning to infer implicit relationships and enrich the knowledge base.
* **Hybrid Deployment Architecture**: Optimized for Streamlit Community Cloud with secure ngrok tunneling to local Triple Stores.

## 🛠️ Technical Stack
* **Language**: Python 3.10+
* **Database**: Apache Fuseki (Triple Store)
* **RDF Framework**: RDFLib & OWL-RL
* **LLM Interface**: Google Gemini API
* **Web Framework**: Streamlit
* **Tunneling**: ngrok

## 📋 Setup & Installation

### 1. Database Configuration
* Ensure **Apache Fuseki** is running locally on port `3030`.
* Create a persistent dataset named `ds`.

### 2. Local Environment
* Clone the repository and install dependencies:
    ```bash
    git clone [https://github.com/your-username/rdf-navigator.git](https://github.com/your-username/rdf-navigator.git)
    pip install -r requirements.txt
    ```
* Create a `.env` file with your `GEMINI_API_KEY`.

