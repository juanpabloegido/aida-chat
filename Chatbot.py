from openai import OpenAI
import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import plotly.express as px
import io
import sys
import json
import logging
from datetime import datetime
from contextlib import redirect_stdout
import db_dtypes  # Añadido para soporte de tipos de datos de BigQuery

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AIDA")

# Set page configuration
st.set_page_config(
    page_title="AIDA - Atida Intelligent Data Assistant",
    page_icon="💊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stTitle {
        color: #2e4d7b;
    }
    .metadata-table {
        font-size: 14px;
    }
    .metadata-table th {
        background-color: #2e4d7b;
        color: white;
        padding: 8px;
    }
    .metadata-table td {
        padding: 8px;
    }
    .debug-info {
        font-family: monospace;
        font-size: 12px;
        padding: 10px;
        background-color: #f0f0f0;
        border-radius: 4px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Create credentials from secrets
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=["https://www.googleapis.com/auth/bigquery"]
)

# Initialize OpenAI client
try:
    client = OpenAI(api_key=st.secrets["openai"]["api_key"])
    has_openai = True
except Exception as e:
    has_openai = False

# BigQuery exploration functions
class BigQueryExplorer:
    def __init__(self, bq_client):
        self.client = bq_client
        self._cache = {}
        self.debug_info = []
        logger.info("Inicializando BigQueryExplorer")
    
    def log_debug(self, message):
        """Add debug message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.debug_info.append(f"[{timestamp}] {message}")
        logger.info(message)
    
    def clear_cache(self):
        """Clear the cache"""
        self._cache = {}
        self.log_debug("Cache limpiado")
    
    def get_datasets(self):
        """Get all datasets in the project"""
        self.log_debug("Obteniendo lista de datasets")
        if 'datasets' not in self._cache:
            try:
                datasets = list(self.client.list_datasets())
                self._cache['datasets'] = datasets
                self.log_debug(f"Encontrados {len(datasets)} datasets")
                for ds in datasets:
                    self.log_debug(f"Dataset encontrado: {ds.dataset_id}")
            except Exception as e:
                error_msg = f"Error al obtener datasets: {str(e)}"
                self.log_debug(error_msg)
                st.error(error_msg)
                return []
        return self._cache['datasets']
    
    def get_tables(self, dataset_id):
        """Get all tables in a dataset"""
        self.log_debug(f"Obteniendo tablas para dataset: {dataset_id}")
        cache_key = f'tables_{dataset_id}'
        if cache_key not in self._cache:
            try:
                tables = list(self.client.list_tables(dataset_id))
                self._cache[cache_key] = tables
                self.log_debug(f"Encontradas {len(tables)} tablas en {dataset_id}")
                for table in tables:
                    self.log_debug(f"Tabla encontrada: {dataset_id}.{table.table_id}")
            except Exception as e:
                error_msg = f"Error al obtener tablas del dataset {dataset_id}: {str(e)}"
                self.log_debug(error_msg)
                st.error(error_msg)
                return []
        return self._cache[cache_key]
    
    def get_table_schema(self, dataset_id, table_id):
        """Get schema of a specific table"""
        self.log_debug(f"Obteniendo esquema para tabla: {dataset_id}.{table_id}")
        cache_key = f'schema_{dataset_id}_{table_id}'
        if cache_key not in self._cache:
            try:
                table_ref = self.client.get_table(f"{dataset_id}.{table_id}")
                self._cache[cache_key] = {
                    'schema': table_ref.schema,
                    'num_rows': table_ref.num_rows,
                    'size_mb': table_ref.num_bytes / 1024 / 1024
                }
                self.log_debug(f"Esquema obtenido: {len(table_ref.schema)} columnas")
            except Exception as e:
                error_msg = f"Error al obtener esquema de {dataset_id}.{table_id}: {str(e)}"
                self.log_debug(error_msg)
                st.error(error_msg)
                return None
        return self._cache[cache_key]
    
    def explore_dataset(self, dataset_id):
        """Explore a dataset and return its metadata"""
        self.log_debug(f"Explorando dataset: {dataset_id}")
        tables = self.get_tables(dataset_id)
        metadata = []
        
        for table in tables:
            schema_info = self.get_table_schema(dataset_id, table.table_id)
            if schema_info:
                metadata.append({
                    'table_id': table.table_id,
                    'num_rows': schema_info['num_rows'],
                    'size_mb': schema_info['size_mb'],
                    'columns': [f"{field.name} ({field.field_type})" for field in schema_info['schema']]
                })
        
        self.log_debug(f"Metadata obtenida para {len(metadata)} tablas")
        return metadata
    
    def display_dataset_explorer(self):
        """Display dataset explorer in Streamlit"""
        datasets = self.get_datasets()
        
        if not datasets:
            st.warning("No se encontraron datasets disponibles")
            return
        
        st.markdown("### 📊 Explorador de Datos")
        
        # Solo botón de refresh
        if st.button("🔄 Refresh", key="refresh_explorer"):
            self.clear_cache()
            st.experimental_rerun()
        
        # Usar session_state para mantener el dataset seleccionado
        if "selected_dataset" not in st.session_state:
            st.session_state.selected_dataset = datasets[0].dataset_id
        
        st.session_state.selected_dataset = st.selectbox(
            "Selecciona un dataset:",
            options=[d.dataset_id for d in datasets],
            key="dataset_selector",
            index=[d.dataset_id for d in datasets].index(st.session_state.selected_dataset)
        )
        
        if st.session_state.selected_dataset:
            self.log_debug(f"Dataset seleccionado: {st.session_state.selected_dataset}")
            metadata = self.explore_dataset(st.session_state.selected_dataset)
            
            if not metadata:
                st.warning(f"No se encontraron tablas en el dataset {st.session_state.selected_dataset}")
                self.log_debug(f"No se encontraron tablas en {st.session_state.selected_dataset}")
            
            # Container para las tablas
            tables_container = st.container()
            with tables_container:
                for table_info in metadata:
                    with st.expander(f"📋 {table_info['table_id']}", expanded=False):
                        st.markdown(f"""
                        **Filas:** {table_info['num_rows']:,}  
                        **Tamaño:** {table_info['size_mb']:.2f} MB
                        
                        **Columnas:**
                        """)
                        for col in table_info['columns']:
                            st.markdown(f"- {col}")
                        
                        preview_key = f"preview_{st.session_state.selected_dataset}_{table_info['table_id']}"
                        if preview_key not in st.session_state:
                            st.session_state[preview_key] = False
                            
                        if st.button(
                            "👁️ Ver preview" if not st.session_state[preview_key] else "🔄 Actualizar preview",
                            key=f"btn_{preview_key}"
                        ):
                            st.session_state[preview_key] = True
                            
                        if st.session_state[preview_key]:
                            preview = self.get_table_preview(st.session_state.selected_dataset, table_info['table_id'])
                            if isinstance(preview, pd.DataFrame):
                                st.dataframe(preview)

    def get_table_preview(self, dataset_id, table_id, limit=5):
        """Get a preview of table data"""
        self.log_debug(f"Obteniendo preview para tabla: {dataset_id}.{table_id}")
        query = f"""
        SELECT *
        FROM `{dataset_id}.{table_id}`
        LIMIT {limit}
        """
        try:
            df = self.client.query(query).to_dataframe()
            self.log_debug(f"Preview obtenido: {len(df)} filas")
            return df
        except Exception as e:
            error_msg = f"Error al obtener preview de {dataset_id}.{table_id}: {str(e)}"
            self.log_debug(error_msg)
            st.error(error_msg)
            return None

with st.sidebar:
    st.image("https://www.atida.com/static/version1741757720/frontend/Interactiv4/mifarmaHyva/es_ES/images/logo.svg", width=200)
    if not has_openai:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    st.divider()
    st.caption("Powered by Atida © 2025")

st.title("🤖 AIDA - Atida Intelligent Data Assistant")
st.caption("Your pharmaceutical data analysis and visualization companion")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "¡Hola! Soy AIDA, tu asistente inteligente de datos de Atida. Puedo ayudarte con consultas de BigQuery, análisis de datos y visualizaciones. ¿En qué puedo ayudarte hoy?"}]

# Initialize BigQuery client with service account
try:
    bq_client = bigquery.Client(
        credentials=credentials,
        project=credentials.project_id
    )
    st.sidebar.success("✅ Conectado a BigQuery")
    # Initialize BigQuery explorer only once
    if "bq_explorer" not in st.session_state:
        st.session_state.bq_explorer = BigQueryExplorer(bq_client)
    bq_explorer = st.session_state.bq_explorer
except Exception as e:
    st.sidebar.error(f"❌ Error de conexión a BigQuery: {str(e)}")
    bq_client = None
    bq_explorer = None

def execute_code(code):
    """Execute Python code and return output"""
    output = io.StringIO()
    with redirect_stdout(output):
        try:
            # Add BigQuery explorer to globals
            globals()['bq_explorer'] = bq_explorer
            exec(code, globals())
            return output.getvalue()
        except Exception as e:
            return f"Error: {str(e)}"

def query_bigquery(query):
    """Execute BigQuery query and return results as DataFrame"""
    try:
        return bq_client.query(query).to_dataframe()
    except Exception as e:
        return f"Error executing query: {str(e)}"

def create_visualization(df, chart_type='line', x=None, y=None):
    """Create visualization using Plotly"""
    if isinstance(df, str):  # If df is an error message
        return df
    
    try:
        if chart_type == 'line':
            fig = px.line(df, x=x, y=y, template="simple_white")
        elif chart_type == 'bar':
            fig = px.bar(df, x=x, y=y, template="simple_white")
        elif chart_type == 'scatter':
            fig = px.scatter(df, x=x, y=y, template="simple_white")
        # Add Atida brand colors
        fig.update_traces(marker_color='#2e4d7b')
        return fig
    except Exception as e:
        return f"Error creating visualization: {str(e)}"

# Add dataset explorer to sidebar
if bq_explorer:
    with st.sidebar:
        st.divider()
        show_explorer = st.button("🔍 Explorar Datasets", key="show_explorer")
        
    # Mostrar explorador fuera del sidebar para evitar conflictos
    if show_explorer:
        st.session_state.bq_explorer.display_dataset_explorer()
    elif "selected_dataset" in st.session_state:  # Mantener el explorador visible si ya estaba
        st.session_state.bq_explorer.display_dataset_explorer()

# Mostrar el historial de mensajes con los resultados
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Procesar el contenido del mensaje
        content = msg["content"]
        if msg["role"] == "assistant" and "<div class='dataframe-result'>" in content:
            # Separar el contenido en partes
            parts = content.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 0:  # Texto regular o DataFrame
                    if "<div class='dataframe-result'>" in part:
                        # Convertir HTML del DataFrame de vuelta a DataFrame
                        df_html = part.split("<div class='dataframe-result'>")[1].split("</div>")[0]
                        df = pd.read_html(df_html)[0]
                        st.dataframe(df)
                    else:
                        st.write(part)
                else:  # Bloques de código
                    lang = part.split("\n")[0]
                    code = "\n".join(part.split("\n")[1:])
                    st.code(code, language=lang.strip())
        else:
            st.write(content)

if prompt := st.chat_input():
    # Check for OpenAI API key
    if not has_openai and not openai_api_key:
        st.info("Por favor, añade tu API key de OpenAI para continuar.")
        st.stop()
    
    # Initialize OpenAI client if using manual key
    if not has_openai:
        client = OpenAI(api_key=openai_api_key)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Enhanced system message
    messages = [
        {"role": "system", "content": f"""You are AIDA (Atida Intelligent Data Assistant), a specialized AI assistant for Atida, a leading European pharmaceutical company. You can:
         1. Query pharmaceutical data from BigQuery
         2. Execute Python code for data analysis
         3. Create visualizations of pharmaceutical trends and metrics
         4. Explore available datasets and tables using bq_explorer
         
         Current available datasets and tables:
         {chr(10).join([f"- Dataset: {d.dataset_id}" + chr(10) + chr(10).join([f"  * {t.table_id}" for t in bq_explorer.get_tables(d.dataset_id)]) for d in bq_explorer.get_datasets()])}
         
         IMPORTANT RULES:
         1. When users ask to see data from a table, ALWAYS respond with a SQL query, not Python code
         2. Always use the full table path with backticks, like: `dataset_name.table_name`
         3. For simple data previews, use: SELECT * FROM `dataset.table` LIMIT X
         4. For specific analyses, use appropriate SQL aggregations and filters
         
         Your personality:
         - Professional but friendly
         - Knowledgeable about pharmaceutical data and healthcare metrics
         - Responds in Spanish by default
         - Always considers data privacy and healthcare regulations
         
         When responding, format your queries as:
         ```sql
         SELECT * FROM `dataset.table` LIMIT 10
         ```
         
         For visualizations after getting data, use:
         ```python
         create_visualization(df, chart_type='bar', x='column1', y='column2')
         ```
         
         Helper functions available:
         - bq_explorer.get_datasets(): List all datasets
         - bq_explorer.get_tables(dataset_id): List tables in a dataset
         - bq_explorer.get_table_schema(dataset_id, table_id): Get table schema"""},
        *st.session_state.messages
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        msg = response.choices[0].message.content
    except Exception as e:
        st.error(f"Error al comunicarse con OpenAI: {str(e)}")
        st.stop()
    
    # Process code blocks in the response
    full_response = []
    with st.chat_message("assistant"):
        code_blocks = msg.split("```")
        for i, block in enumerate(code_blocks):
            if i % 2 == 0:  # Regular text
                st.write(block)
                full_response.append(block)
            else:  # Code block
                lang = block.split("\n")[0]
                code = "\n".join(block.split("\n")[1:])
                st.code(code, language=lang.strip())
                full_response.append(f"```{lang}\n{code}\n```")
                
                if lang.strip() == "sql" and bq_client:
                    results = query_bigquery(code)
                    if isinstance(results, pd.DataFrame):
                        st.dataframe(results)
                        # Guardar una representación del DataFrame en el mensaje
                        df_html = f"<div class='dataframe-result'>{results.to_html()}</div>"
                        full_response.append(df_html)
                    else:
                        st.error(results)
                        full_response.append(f"Error: {results}")
                
                elif lang.strip() == "python":
                    output = execute_code(code)
                    if output:
                        st.text(output)
                        full_response.append(f"```\n{output}\n```")

    # Guardar el mensaje completo incluyendo los resultados
    st.session_state.messages.append({"role": "assistant", "content": "\n".join(full_response)})
