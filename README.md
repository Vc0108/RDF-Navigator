# 🕸️ Unified RDF Navigator: AI-Driven Knowledge Graph Explorer

An advanced Semantic Web application designed to transform heterogeneous tabular data into a structured **Knowledge Graph (RDF)** and enable natural language exploration via **Generative AI**.

---

## 🔗 Live Application
**Access the deployed app here:** [https://rdf-navigator-atgyrgyiifxrfyedxxnvkc.streamlit.app/](https://rdf-navigator-atgyrgyiifxrfyedxxnvkc.streamlit.app/)

---

## 🚀 Project Overview
Traditional relational databases often struggle to represent complex, multi-layered relationships between entities. This project addresses that challenge by converting standard datasets (CSV, Excel, JSON) into **RDF triples** (Subject-Predicate-Object), creating a machine-readable network of interconnected data. 

By integrating the **Google Gemini Pro** model, the application allows users to bypass complex SPARQL syntax and query the database using plain English.

## ✨ Key Technical Features
* **Dynamic Triple Generation**: Automatically maps tabular columns to RDF predicates and row values to unique URIs (Uniform Resource Identifiers) to ensure data integrity.
* **AI-Powered SPARQL Synthesis**: Utilizes Retrieval-Augmented Generation (RAG) to translate natural language questions into valid SPARQL 1.1 queries in real-time.
* **Interactive Visual Analytics**: Employs force-directed graph algorithms via **PyVis** to visualize data clusters and relationship paths dynamically.
* **Semantic Reasoning Engine**: Built-in support for **RDFS reasoning** (via OWL-RL) to infer implicit relationships and enrich the knowledge base automatically.
* **Hybrid Cloud Architecture**: Optimized for **Streamlit Community Cloud** with secure **ngrok** tunneling to interact with local private Triple Stores.

## 🛠️ Technical Stack
* **Core Language**: Python 3.10+
* **Triple Store**: Apache Fuseki
* **Semantic Framework**: RDFLib & OWL-RL
* **LLM Interface**: Google Gemini API
* **Web Framework**: Streamlit
* **Tunneling Utility**: ngrok

## 📋 Setup & Installation

### 1. Database Configuration
* Start your **Apache Fuseki** server on port `3030`.
* Create a persistent dataset named `ds`.

### 2. Deployment & Tunneling
Since the frontend is hosted on Streamlit Cloud and the database is local, use **ngrok** to bridge the connection:
```bash
ngrok http 3030
