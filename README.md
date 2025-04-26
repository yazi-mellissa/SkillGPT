# Skills Extraction API
A FastAPI application that provides endpoints for searching skills using various AI models including OpenAI, Google Gemini, with vector search capabilities using Milvus/Zilliz.

## Features

- Multiple skill search endpoints using different AI models:
  - Vector search using Milvus/Zilliz
  - OpenAI model-based skill extraction
  - Google Gemini AI-based skill extraction
  - BERT model-based skill extraction
- Response caching with GPTCache
- Docker and Docker Compose support for easy deployment
- Structured project layout following best practices

## Project Structure

```
/your_project/
│
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI app startup
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py         # API endpoints
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── bert_service.py   # BERT model service
│   │   ├── gemini_service.py # Gemini AI service
│   │   ├── openai_service.py # OpenAI service
│   │   ├── milvus_service.py # Milvus/Zilliz service
│   │   ├── gpt_cache.py      # GPTCache integration
│   │   ├── memory_manager.py # Zilliz Cloud management
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py         # Configuration loading
│   │   ├── logging.py        # Logging configuration
│
├── .env                      # Environment variables
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker Compose configuration
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
```

## Prerequisites

- Python 3.11+
- Docker and Docker Compose (for containerized deployment)
- API keys for OpenAI and Google Gemini
- Zilliz Cloud account (for vector database)

## Environment Variables

Copy the `.env.example` file to `.env` and fill in your API keys and configuration:

```
# API Server Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=

# API Keys
OPEN_API_KEY=
GEMINI_API_KEY=

# Caching Configuration
USE_GPTCACHE=true

# Zilliz/Milvus Configuration
ZILLIZ_URI=
ZILLIZ_TOKEN=
WIPE_MILVUS_ON_START=false

# Model Configuration
GEMINI_MODEL_NAME=
OPENAI_MODEL_NAME=
```

## Installation and Running

### Using Docker Compose (Recommended)

1. Build and start the containers:
   ```
   docker-compose up -d
   ```

2. Access the API at http://localhost:8000

### Manual Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   uvicorn app.main:app --reload
   ```

## API Endpoints

- `/search?query=...` - Search skills using vector embeddings in Milvus/Zilliz
- `/search_openai?query=...` - Extract skills using OpenAI models
- `/search_gemini?query=...` - Extract skills using Google Gemini AI
- `/api/health` - Health check endpoint

## Development

To run the application in development mode:

```
uvicorn app.main:app --reload
```

This will automatically reload the application when changes are detected.
