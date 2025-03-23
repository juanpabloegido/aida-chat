# AIDA - Atida Intelligent Data Assistant

AIDA es un chatbot inteligente diseñado para Atida que permite:

- Consultar datos de BigQuery
- Analizar datos farmacéuticos
- Crear visualizaciones
- Explorar datasets y tablas

## Instalación

```bash
pip install -r requirements.txt
```

## Configuración

Crea un archivo `.streamlit/secrets.toml` con las siguientes credenciales:

```toml
[gcp_service_account]
# Credenciales de BigQuery

[openai]
api_key = "tu-api-key"
```

## Run it locally

```sh
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run Chatbot.py
```

## Uso

```bash
streamlit run Chatbot.py
```
