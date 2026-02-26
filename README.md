# 🕸️ RDF Navigator v4

An industry-ready RDF Knowledge Graph tool — no server required.  
Built with **rdflib + Oxigraph + Streamlit + Groq AI**.

---

## 🌐 Live Demo

👉 **[https://rdf-navigator-e7ljskxcshauwwj5bfvrmf.streamlit.app/](https://rdf-navigator-e7ljskxcshauwwj5bfvrmf.streamlit.app/)**

> No installation needed — open the link and start exploring!

---

## ✨ Features

| Feature | Description |
|---|---|
| 📥 Smart Import | CSV, XLSX, JSON → RDF Turtle automatically |
| 🔍 Graph Explorer | Browse nodes, navigate relationships |
| 📊 SPARQL Suite | Predefined + custom SPARQL queries with export |
| 🛤️ Multi-Hop Paths | Find how any two resources connect (1–4 hops) |
| 🤖 AI Assistant | Ask questions in plain English → auto SPARQL (Groq — Free) |
| 🕸️ Graph View | Interactive PyVis visualization with dynamic colors |
| 🔀 RDF Diff | Snapshot & compare graph changes over time |
| 🧬 Auto-Ontology | Generate OWL ontology from your data automatically |
| 🧠 Reasoning | Apply OWL/RDFS rules to infer new facts |
| ⬇️ Export | Download any result as CSV or Excel |
| 📊 Stats Dashboard | Live graph metrics — triples, classes, predicates |

---

## 🚀 Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/rdf-navigator.git
cd rdf-navigator

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key (optional, for AI features)
cp .env.example .env
# Edit .env and add: GROQ_API_KEY=your_key_here

# 5. Run
streamlit run rdf_navigator_v4.py
```

---

## ☁️ Deploy to Streamlit Community Cloud (Free Public Link)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "RDF Navigator v4"
git remote add origin https://github.com/YOUR_USERNAME/rdf-navigator.git
git push -u origin main
```

### Step 2 — Deploy
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repo → branch: `main` → file: `rdf_navigator_v4.py`
5. Click **"Advanced settings"** → add secret:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
6. Click **Deploy**

> Get your **free** Groq API key at [console.groq.com](https://console.groq.com) — no credit card needed!

---

## 📁 Project Structure

```
rdf-navigator/
├── rdf_navigator_v4.py        ← Main app (production)
├── requirements.txt           ← Python dependencies
├── .env.example               ← Template for env vars
├── .gitignore                 ← Excludes .env and cache files
├── README.md
└── .streamlit/
    └── config.toml            ← Blue & White theme config
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Optional | Free Groq key for AI — get at console.groq.com |

---

## 🏗️ Architecture

```
User Browser
    ↓
Streamlit (Python)
    ↓
OxigraphStore (embedded in-memory)
    ↓ (rdflib Graph synced)
RDFNavigator / SPARQL Engine
    ↓
Groq AI (NL → SPARQL)
```

No Apache Fuseki server needed. All data is embedded in the running process.

> **Note on Data Persistence:** Data resets on app restart (Streamlit Cloud).
> Export your graph as TTL using the sidebar button and re-import on the next session.

---

## 📦 Dependencies

- [rdflib](https://rdflib.readthedocs.io/) — RDF graph engine
- [pyoxigraph](https://pyoxigraph.readthedocs.io/) — Fast embedded triple store
- [pyvis](https://pyvis.readthedocs.io/) — Interactive graph visualization
- [owlrl](https://owl-rl.readthedocs.io/) — OWL/RDFS reasoning
- [groq](https://console.groq.com/) — Free AI for NL→SPARQL (llama3-70b)
- [streamlit](https://streamlit.io/) — Web UI framework
- [openpyxl](https://openpyxl.readthedocs.io/) — Excel export

---

## 👨‍💻 Built During Internship

This project was built as part of an internship, evolving from a basic RDF navigator
to a full production-ready knowledge graph tool with AI capabilities.
