import streamlit as st
import pandas as pd
from rdflib import Graph, URIRef, Literal
from rdflib.plugins.stores.sparqlstore import SPARQLStore
import requests
import io
import os
import re
from datetime import datetime
import json
import hashlib
import streamlit.components.v1 as components
from pyvis.network import Network
import google.generativeai as genai
from dotenv import load_dotenv
import owlrl 

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration & Page Setup ---
st.set_page_config(
    page_title="Unified RDF Navigator",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🕸️"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container { margin-top: -2em; }
    .stDeployButton {display:none;}
    div[data-testid="stExpander"] div[role="button"] p { font-weight: bold; }
    .stAlert { padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# --- Classes & Logic ---

class FusekiConnector:
    """Handles communication with the Fuseki Triple Store"""
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")
        self.data_url = f"{self.base_url}/data"
        self.sparql_url = f"{self.base_url}/sparql"
        self.update_url = f"{self.base_url}/update"

    def check_connection(self):
        try:
            requests.get(self.sparql_url, timeout=2)
            return True
        except:
            return False

    def clear_all(self):
        try:
            return requests.post(self.update_url, data={"update": "CLEAR ALL"})
        except Exception as e:
            return e

    def upload_ttl(self, ttl_data):
        headers = {"Content-Type": "text/turtle"}
        return requests.post(self.data_url, data=ttl_data.encode("utf-8"), headers=headers)

class FileManager:
    """Manages file metadata persistence"""
    def __init__(self, storage_file="uploaded_files.json"):
        self.storage_file = storage_file
        if 'file_registry' not in st.session_state:
            self.load_files()

    def load_files(self):
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    st.session_state.file_registry = json.load(f)
            else:
                st.session_state.file_registry = []
        except:
            st.session_state.file_registry = []

    def save_files(self):
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(st.session_state.file_registry, f, indent=2)
        except Exception as e:
            st.error(f"Error saving file metadata: {e}")

    def add_file(self, filename, ttl_data, triple_count, file_size, namespace):
        file_hash = hashlib.md5(f"{filename}{datetime.now()}".encode()).hexdigest()
        file_info = {
            'id': file_hash,
            'filename': filename,
            'upload_time': datetime.now().isoformat(),
            'triple_count': triple_count or 0,
            'file_size': file_size,
            'ttl_data': ttl_data,
            'namespace': namespace
        }
        st.session_state.file_registry.append(file_info)
        self.save_files()
        return file_hash

    def get_files(self):
        return sorted(st.session_state.file_registry, key=lambda x: x['upload_time'], reverse=True)

class DataToRDFConverter:
    """
    Converts tabular data to RDF Turtle format.
    Includes logic to handle datasets without unique IDs by appending row numbers.
    """
    def __init__(self, namespace, prefix="ex"):
        self.namespace = namespace
        self.prefix = prefix

    def clean_text(self, text):
        if pd.isna(text) or text is None: return ""
        return str(text).strip().replace('\\', '\\\\').replace('"', '\\"')

    def create_uri(self, entity_type, identifier, row_index):
        # FIX: Append row_index to ensure uniqueness for datasets like EV Sales
        clean_id = re.sub(r'[^a-zA-Z0-9_]', '_', str(identifier))
        return f"{self.prefix}:{entity_type}_{clean_id}_{row_index}"

    def load_file_to_df(self, uploaded_file):
        fname = uploaded_file.name.lower()
        if fname.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif fname.endswith('.xlsx') or fname.endswith('.xls'):
            return pd.read_excel(uploaded_file)
        elif fname.endswith('.json'):
            return pd.read_json(uploaded_file)
        return None

    def convert_to_ttl(self, df, filename, id_col_name, ignore_cols=[], source_file_id=None):
        output = io.StringIO()
        output.write(f"@prefix {self.prefix}: <{self.namespace}> .\n")
        output.write("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n")
        output.write("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\n")

        triple_count = 0
        
        # Reset index to get clean row numbers
        df = df.reset_index(drop=True)
        
        for idx, row in df.iterrows():
            if pd.isna(row[id_col_name]) or not str(row[id_col_name]).strip(): continue
            
            subj_val = str(row[id_col_name]).strip()
            entity_type = re.sub(r'[^a-zA-Z0-9]', '', id_col_name.title())
            
            # PASS ROW INDEX HERE to create unique IDs (e.g., Australia_0, Australia_1)
            subj_uri = self.create_uri(entity_type, subj_val, idx)
            
            output.write(f"{subj_uri} a {self.prefix}:{entity_type} ;\n")
            triple_count += 1
            
            if source_file_id:
                output.write(f"    {self.prefix}:sourceFile \"{source_file_id}\" ;\n")
                triple_count += 1

            for col in df.columns:
                if col == id_col_name or col in ignore_cols: continue
                if pd.isna(row[col]) or str(row[col]).strip() == '': continue
                
                clean_pred = re.sub(r'[^a-zA-Z0-9_]', '_', col.strip())
                pred = f"{self.prefix}:{clean_pred}"
                val = row[col]
                
                if isinstance(val, (int, float)):
                    output.write(f"    {pred} \"{val}\"^^xsd:{'float' if isinstance(val, float) else 'integer'} ;\n")
                else:
                    output.write(f"    {pred} \"{self.clean_text(val)}\" ;\n")
                triple_count += 1
            
            output.seek(output.tell() - 2)
            output.write(" .\n")
            
        return output.getvalue(), triple_count

class RDFNavigator:
    def __init__(self, graph):
        self.graph = graph

    def shorten_uri(self, uri):
        if not uri: return ""
        uri = str(uri)
        if '#' in uri: return uri.split('#')[-1]
        if '/' in uri: return uri.split('/')[-1]
        return uri

    def get_resource_triples(self, resource_uri):
        try:
            uri_ref = URIRef(resource_uri)
            triples = []
            for s, p, o in self.graph.triples((uri_ref, None, None)):
                triples.append(('subject', s, p, o)) 
            for s, p, o in self.graph.triples((None, None, uri_ref)):
                triples.append(('object', s, p, o)) 
            return triples
        except Exception as e:
            return []

    def execute_sparql(self, query):
        try:
            results = self.graph.query(query)
            return list(results), None
        except Exception as e:
            return [], str(e)
            
    def search_nodes(self, keyword):
        """Finds URIs that contain the keyword (Case Insensitive)"""
        try:
            query = f"""
            SELECT DISTINCT ?s WHERE {{
                ?s ?p ?o .
                FILTER (regex(str(?s), "{keyword}", "i"))
            }} LIMIT 20
            """
            results = self.graph.query(query)
            return [str(row[0]) for row in results]
        except:
            return []

class GraphRAG:
    """Handles AI-to-SPARQL conversion with Auto-Model-Detection"""
    def __init__(self, api_key, graph):
        self.api_key = api_key
        self.graph = graph
        self.model_name = "Unknown"
        self.model = None

        if api_key:
            genai.configure(api_key=api_key)
            self.model_name = self.find_working_model()
            self.model = genai.GenerativeModel(self.model_name)
    
    def find_working_model(self):
        """
        AUTO-FIX: Scans the user's library to find ANY working model.
        This bypasses version mismatch errors (404/403).
        """
        try:
            all_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            preferences = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-1.0-pro', 'models/gemini-pro']
            for pref in preferences:
                if pref in all_models:
                    return pref
            if all_models: return all_models[0]
            return "gemini-pro"
        except Exception as e:
            return "gemini-pro"

    def get_schema_summary(self):
        query = "SELECT DISTINCT ?type ?p WHERE { ?s a ?type . ?s ?p ?o . } LIMIT 100"
        results = self.graph.query(query)
        schema = []
        for row in results:
            schema.append(f"Type: {row.type} has property {row.p}")
        return "\n".join(schema)

    def ask_question(self, question, chat_history=[]):
        if not self.api_key:
            return "NO_KEY", "Please enter a Google Gemini API Key in the sidebar."

        schema_context = self.get_schema_summary()
        
        # --- NEW: Build Conversation Context ---
        history_context = ""
        for msg in chat_history[-5:]:
            role = "USER" if msg["role"] == "user" else "AI"
            content = msg["content"]
            history_context += f"{role}: {content}\n"

        prompt = f"""
        You are an expert in SPARQL. Translate the following natural language question into a valid SPARQL query.
        
        GRAPH SCHEMA:
        {schema_context}
        
        PREVIOUS CONVERSATION:
        {history_context}
        
        CURRENT QUESTION: "{question}"
        
        RULES:
        1. Return ONLY the SPARQL query code.
        2. Use the prefixes found in the schema.
        3. If the question implies following up on previous results, use the context.
        4. No Markdown blocks.
        """
        try:
            response = self.model.generate_content(prompt)
            sparql_query = response.text.replace("```sparql", "").replace("```", "").strip()
            return "SUCCESS", sparql_query
        except Exception as e:
            return "ERROR", str(e)

class ReasoningEngine:
    def __init__(self, fuseki_connector):
        self.connector = fuseki_connector

    def run_inference(self, ontology_ttl, max_triples=5000):
        try:
            full_data_query = f"CONSTRUCT {{ ?s ?p ?o }} WHERE {{ ?s ?p ?o }} LIMIT {max_triples}"
            response = requests.get(self.connector.sparql_url, params={'query': full_data_query})
            
            g = Graph()
            g.parse(data=response.text, format="turtle")
            initial_count = len(g)
            
            if initial_count == 0:
                return False, "No data found in Fuseki to reason over."

            g.parse(data=ontology_ttl, format="turtle")
            owlrl.DeductiveClosure(owlrl.RDFS_Semantics).expand(g)
            
            final_count = len(g)
            new_triples = final_count - initial_count

            if new_triples > 0:
                inferred_ttl = g.serialize(format="turtle")
                self.connector.upload_ttl(inferred_ttl)
                return True, new_triples
            else:
                return True, 0

        except Exception as e:
            return False, str(e)


# --- Initialization ---
if 'current_resource_uri' not in st.session_state:
    st.session_state.current_resource_uri = None
if 'preview_dfs' not in st.session_state:
    st.session_state.preview_dfs = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

file_manager = FileManager()

# --- Sidebar ---
st.sidebar.title("⚙️ Configuration")

# 1. Connection
st.sidebar.subheader("1. Connection")
def_store_url = "http://localhost:3030/ds"
store_url = st.sidebar.text_input("Fuseki URL", value=def_store_url)
connector = FusekiConnector(store_url)

if connector.check_connection():
    st.sidebar.success("🟢 Fuseki Online")
    store = SPARQLStore(connector.sparql_url)
    triplestore_graph = Graph(store=store)
    reasoner = ReasoningEngine(connector)
else:
    st.sidebar.error("🔴 Fuseki Offline")
    triplestore_graph = None
    reasoner = None

st.sidebar.divider()

# 2. AI Settings
st.sidebar.subheader("2. AI Assistant")
env_key = os.getenv("GEMINI_API_KEY")
if env_key:
    st.sidebar.success("🔑 API Key Loaded from .env")
    gemini_key = env_key
else:
    gemini_key = st.sidebar.text_input("Google Gemini API Key", type="password", help="Get key at aistudio.google.com")

rag_engine = None
if triplestore_graph:
    rag_engine = GraphRAG(gemini_key, triplestore_graph)
    if rag_engine.model_name != "Unknown":
        st.sidebar.info(f"✅ AI Connected: {rag_engine.model_name}")
    else:
        st.sidebar.warning("⚠️ AI Model not found. Check API Key.")

st.sidebar.divider()

# 3. Namespace Configuration
st.sidebar.subheader("3. Data Context")

with st.sidebar.expander("📝 View Examples"):
    st.markdown("""
    **Namespace URI:** The unique "web address" for your data entities.
    * *Format:* `http://{domain}/{project}#`
    * *Example:* `http://mycompany.com/finance#`
    
    **Prefix:** A short nickname for the namespace.
    * *Example:* `fin` (for finance)
    """)

custom_ns = st.sidebar.text_input("Namespace URI", value="http://example.org/data#")
custom_prefix = st.sidebar.text_input("Prefix", value="ex")

if not custom_ns.endswith(('#', '/')): custom_ns += '#'
converter = DataToRDFConverter(namespace=custom_ns, prefix=custom_prefix)

st.sidebar.divider()

# 4. Upload
st.sidebar.subheader("4. Smart Import")
uploaded_files = st.sidebar.file_uploader("Upload Data", type=["csv", "xlsx", "xls", "json"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        st.sidebar.markdown(f"---")
        st.sidebar.markdown(f"**📄 `{file.name}`**")
        if file.name not in st.session_state.preview_dfs:
            try:
                file.seek(0)
                st.session_state.preview_dfs[file.name] = converter.load_file_to_df(file)
            except: pass
        
        df = st.session_state.preview_dfs.get(file.name)
        if df is not None:
            default_idx = 0
            for i, col in enumerate(df.columns):
                if 'id' in col.lower() or 'code' in col.lower() or 'key' in col.lower(): 
                    default_idx = i
            
            id_col = st.sidebar.selectbox(
                f"Subject Column", 
                options=df.columns, 
                index=default_idx, 
                key=f"id_{file.name}", 
                help="Select a UNIQUE column (e.g. ID, Name). Do not select Age/Gender!"
            )

            if id_col.lower() in ['age', 'gender', 'sex', 'pclass', 'survived', 'class']:
                st.sidebar.warning(f"⚠️ Warning: '{id_col}' is NOT unique! This will merge different people into one node. Please select a Unique ID (like Name or ID).")

            ignore_cols = st.sidebar.multiselect(f"Ignore Columns", options=[c for c in df.columns if c != id_col], key=f"ignore_{file.name}")
            
            if st.sidebar.button(f"🚀 Import", key=f"btn_{file.name}"):
                try:
                    temp_hash = hashlib.md5(file.name.encode()).hexdigest()
                    ttl_data, count = converter.convert_to_ttl(df, file.name, id_col, ignore_cols, temp_hash)
                    resp = connector.upload_ttl(ttl_data)
                    if resp.status_code in [200, 204]:
                        file_manager.add_file(file.name, ttl_data, count, file.size, custom_ns)
                        st.toast(f"✅ Imported!")
                        del st.session_state.preview_dfs[file.name]
                        st.rerun()
                    else: st.sidebar.error(f"Error: {resp.text}")
                except Exception as e: st.sidebar.error(f"Error: {str(e)}")

# 5. Reset
st.sidebar.divider()
if st.sidebar.button("🗑️ Reset Database", type="primary"):
    if connector.clear_all().status_code == 200:
        st.session_state.file_registry = []
        file_manager.save_files()
        st.session_state.preview_dfs = {} 
        st.session_state.current_resource_uri = None
        st.success("Cleared.")
        st.rerun()

# --- Main Content ---
st.title("🕸️ Unified RDF Navigator")
if triplestore_graph is None:
    st.warning("Please ensure Apache Fuseki is running.")
    st.stop()
navigator = RDFNavigator(triplestore_graph)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Explorer", "🤖 Ask & Query", "🕸️ Graph", "📁 Files", "🧠 Reasoning"])

# --- TAB 1: Explorer ---
with tab1:
    st.markdown("##### Browse your Knowledge Graph")
    search_term = st.text_input("🔎 Search by Name (e.g., 'Arab', 'United')", placeholder="Type a keyword to find URIs...")
    
    if search_term:
        found_uris = navigator.search_nodes(search_term)
        if found_uris:
            st.success(f"Found {len(found_uris)} matches:")
            selected_uri = st.selectbox("Select a Result:", options=found_uris, format_func=lambda x: navigator.shorten_uri(x))
            if st.button("Go to Selected"):
                st.session_state.current_resource_uri = selected_uri
                st.rerun()
        else:
            st.warning("No matches found.")

    st.divider()

    col1, col2 = st.columns([3, 1])
    with col1: 
        current_val = st.session_state.current_resource_uri if st.session_state.current_resource_uri else ""
        search_input = st.text_input("Or Enter Full URI", value=current_val)
    with col2:
        st.write(""); st.write("")
        if st.button("🎲 Random Node"):
            res, _ = navigator.execute_sparql("SELECT ?s WHERE { ?s ?p ?o } LIMIT 1 OFFSET 0")
            if res:
                st.session_state.current_resource_uri = str(res[0][0])
                st.rerun()

    if st.button("Go") or (search_input and search_input != st.session_state.current_resource_uri):
        st.session_state.current_resource_uri = search_input
        st.rerun()

    if st.session_state.current_resource_uri:
        uri = st.session_state.current_resource_uri
        triples = navigator.get_resource_triples(uri)
        st.divider()
        st.subheader(f"📌 {navigator.shorten_uri(uri)}")
        st.caption(f"Full URI: {uri}")
        
        if triples:
            outgoing = [t for t in triples if t[0] == 'subject']
            incoming = [t for t in triples if t[0] == 'object']
            c1, c2 = st.columns(2)
            with c1:
                st.info(f"➡️ Outgoing ({len(outgoing)})")
                if outgoing:
                    df_out = pd.DataFrame([(navigator.shorten_uri(t[2]), navigator.shorten_uri(t[3]), t[3]) for t in outgoing], columns=["Rel", "Val", "FullURI"])
                    st.dataframe(df_out[["Rel", "Val"]], use_container_width=True)
                    navs = [r.FullURI for r in df_out.itertuples() if isinstance(r.FullURI, URIRef)]
                    if navs:
                        t = st.selectbox("Visit:", options=navs, key="n_out", format_func=lambda x: navigator.shorten_uri(x))
                        if st.button("Go", key="b_out"): st.session_state.current_resource_uri = str(t); st.rerun()
            with c2:
                st.success(f"⬅️ Incoming ({len(incoming)})")
                if incoming:
                    df_in = pd.DataFrame([(navigator.shorten_uri(t[1]), navigator.shorten_uri(t[2]), t[1]) for t in incoming], columns=["Src", "Rel", "FullURI"])
                    st.dataframe(df_in[["Src", "Rel"]], use_container_width=True)
                    t_in = st.selectbox("Visit:", options=df_in['FullURI'].unique(), key="n_in", format_func=lambda x: navigator.shorten_uri(x))
                    if st.button("Go", key="b_in"): st.session_state.current_resource_uri = str(t_in); st.rerun()
        else:
            st.warning("No data found for this URI.")

# --- TAB 2: Ask & Query ---
with tab2:
    st.header("🤖 Ask & Query")
    
    col_c1, col_c2 = st.columns([1, 4])
    with col_c1:
        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    with col_c2:
        if st.session_state.chat_history:
            report_text = f"Unified RDF Navigator - Session Report\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + "="*50 + "\n\n"
            for msg in st.session_state.chat_history:
                report_text += f"[{msg['role'].upper()}]: {msg['content']}\n"
                if 'sql' in msg: report_text += f"SPARQL QUERY:\n{msg['sql']}\n"
                report_text += "-"*30 + "\n"
            st.download_button("📥 Download Report", report_text, file_name=f"rdf_session_report_{datetime.now().strftime('%H%M')}.txt")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sql" in msg: st.code(msg["sql"], language="sparql")
            if "df" in msg: st.dataframe(msg["df"])
            
    if prompt := st.chat_input("Ask a question (e.g., 'Show me sales in Australia')"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            if not rag_engine or not gemini_key: st.error("⚠️ Configure Gemini API Key first.")
            else:
                with st.spinner("Thinking..."):
                    status, sparql_code = rag_engine.ask_question(prompt, st.session_state.chat_history)
                    if status == "SUCCESS":
                        st.markdown("**Generated Query:**"); st.code(sparql_code, language="sparql")
                        res, err = navigator.execute_sparql(sparql_code)
                        if err: 
                            st.error(f"Error: {err}")
                            st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {err}"})
                        elif res:
                            cols = [str(v) for v in res[0].labels]
                            data = [[str(i) for i in r] for r in res]
                            df_res = pd.DataFrame(data, columns=cols)
                            st.success(f"Found {len(df_res)} results.")
                            st.dataframe(df_res)
                            st.session_state.chat_history.append({"role": "assistant", "content": f"Found {len(df_res)} results.", "sql": sparql_code, "df": df_res})
                        else:
                            st.warning("No results.")
                            st.session_state.chat_history.append({"role": "assistant", "content": "No results found.", "sql": sparql_code})
                    else: st.error(f"AI Error: {sparql_code}")

    st.divider()
    with st.expander("⚡ Advanced: Manual SPARQL Editor"):
        st.markdown("Write and run raw SPARQL queries here.")
        q_text = st.text_area("SPARQL Code", height=150, value="SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 25")
        if st.button("Run Manual Query"):
            res, err = navigator.execute_sparql(q_text)
            if err: st.error(err)
            elif res:
                cols = [str(v) for v in res[0].labels] if res else []
                data = [[str(i) for i in r] for r in res]
                st.dataframe(pd.DataFrame(data, columns=cols), use_container_width=True)
            else: st.info("No results.")

# --- TAB 3: Dynamic Graph (SMART COLORS) ---
with tab3:
    if not st.session_state.current_resource_uri: 
        st.info("Select a resource in Explorer first.")
    else:
        # Layout Controls
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1: show_lit = st.checkbox("Show Data (Literals)", True)
        with c2: hier = st.checkbox("Tree Layout", False)
        
        # Get Data
        triples = navigator.get_resource_triples(st.session_state.current_resource_uri)
        
        # --- DYNAMIC COLOR LOGIC ---
        # 1. Identify all unique Predicates (Relationship types) in this view
        unique_preds = list(set([navigator.shorten_uri(t[2]) for t in triples if t[0] == 'subject'] + 
                                [navigator.shorten_uri(t[1]) for t in triples if t[0] == 'object']))
        
        # 2. Define a Palette (12 Distinct Colors)
        palette = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff"]
        
        # 3. Assign Color to Predicate
        color_map = {}
        for i, pred in enumerate(unique_preds):
            color_map[pred] = palette[i % len(palette)]
            
        # 4. Display Legend
        with c3:
            st.caption("Dynamic Legend:")
            legend_html = ""
            for pred, color in color_map.items():
                legend_html += f"<span style='background-color:{color}; padding:2px 8px; border-radius:4px; color:white; margin-right:5px; font-size:12px;'>{pred}</span>"
            st.markdown(legend_html, unsafe_allow_html=True)
        
        # Initialize Graph
        net = Network(height="600px", width="100%", bgcolor="white", font_color="black")
        curr = st.session_state.current_resource_uri
        short_curr = navigator.shorten_uri(curr)
        
        # Central Node (Black for contrast)
        net.add_node(curr, label=short_curr, color="#000000", size=30, title=f"Center: {short_curr}")

        cnt = 0
        for t_type, s, p, o in triples:
            if cnt > 100: break 
            
            lbl = navigator.shorten_uri(p)
            node_color = color_map.get(lbl, "#999999") # Get dynamic color
            node_shape = "dot"
            
            if t_type == 'object': 
                # Incoming edges
                net.add_node(str(s), label=navigator.shorten_uri(s), color=node_color, size=20)
                net.add_edge(str(s), curr, label=lbl, color=node_color)
            else:
                # Outgoing edges
                if isinstance(o, Literal):
                    if not show_lit: continue
                    node_shape = "box"
                
                net.add_node(str(o), label=navigator.shorten_uri(o)[:20], color=node_color, shape=node_shape, size=20)
                net.add_edge(curr, str(o), label=lbl, color=node_color)
            
            cnt += 1

        ops = '{"layout": {"hierarchical": {"enabled": true, "direction": "UD", "sortMethod": "directed"}}}' if hier else '{"physics": {"forceAtlas2Based": {"gravitationalConstant": -50, "springLength": 100}, "minVelocity": 0.75, "solver": "forceAtlas2Based"}}'
        net.set_options(ops)
        
        try:
            path = "graph.html"
            net.save_graph(path)
            with open(path, 'r', encoding='utf-8') as f:
                html_data = f.read()
            components.html(html_data, height=600)
        except Exception as e:
            st.error(f"Graph Error: {e}")

# --- TAB 4: Files ---
with tab4:
    files = file_manager.get_files()
    st.dataframe(pd.DataFrame(files)[['filename', 'triple_count', 'upload_time']], use_container_width=True)

# --- TAB 5: Reasoning ---
with tab5:
    st.header("🧠 Semantic Reasoning")
    st.caption("Apply logical rules to infer new facts.")
    
    col_r1, col_r2 = st.columns([1, 1])
    
    with col_r1:
        st.subheader("1. Define Rules (Ontology)")
        default_rules = """@prefix ex: <http://example.org/data#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

# Example for EV Data
ex:Oceania rdfs:subClassOf ex:GlobalMarket .
ex:North_America rdfs:subClassOf ex:GlobalMarket .
ex:mass_market rdfs:subClassOf ex:PassengerVehicle .
"""
        rules_input = st.text_area("Turtle Rules (TTL)", value=default_rules, height=300)
    
    with col_r2:
        st.subheader("2. Run Inference")
        if st.button("🚀 Run Reasoner"):
            if not reasoner:
                st.error("Fuseki is offline.")
            else:
                with st.spinner("Reasoning in progress..."):
                    success, result = reasoner.run_inference(rules_input)
                    if success:
                        if result > 0:
                            st.success(f"✅ Success! Inferred {result} new facts.")
                            st.balloons()
                            st.divider()
                            st.subheader("👀 Preview of Inferred Facts")
                            
                            check_query = """
                            PREFIX ex: <http://example.org/data#>
                            SELECT ?Subject ?NewType WHERE { 
                                ?Subject a ?NewType .
                                FILTER (?NewType != ex:Entity) 
                            } LIMIT 20
                            """
                            res, err = navigator.execute_sparql(check_query)
                            if res:
                                cols = [str(v) for v in res[0].labels]
                                data = [[navigator.shorten_uri(str(i)) for i in r] for r in res]
                                st.dataframe(pd.DataFrame(data, columns=cols), use_container_width=True)
                        else:
                            st.warning("✅ Success, but no new facts were inferred.")
                    else:
                        st.error(f"Reasoning Failed: {result}")