# Simple RAG Application

A simple Retrieval-Augmented Generation (RAG) application built with LangChain, PostgreSQL (with pgvector), and Docker Compose.

## Features

- **Natural Language Queries**: Ask questions in natural language
- **Vector Search**: Uses pgvector for efficient similarity search
- **Document Management**: Add and retrieve documents
- **RESTful API**: FastAPI-based REST endpoints
- **Docker Compose**: Easy deployment with all dependencies

## Architecture

```
User Query → FastAPI → LangChain → PostgreSQL/pgvector → OpenAI LLM → Response
```

## Prerequisites

- Docker and Docker Compose
- OpenAI API key


## API Endpoints

### Query Documents
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is Python?"}'
```

### Add Document
```bash
curl -X POST "http://localhost:8000/add-document" \
     -H "Content-Type: application/json" \
     -d '{"content": "Your document content here", "metadata": {"category": "example"}}'
```

### List Documents
```bash
curl "http://localhost:8000/documents"
```

### Health Check
```bash
curl "http://localhost:8000/health"
```

## Example Queries

Try these natural language queries:
- "What is Python?"
- "Tell me about machine learning"
- "How does Docker work?"
- "What is PostgreSQL?"
- "Explain LangChain"



## How It Works

1. **Document Storage**: Documents are stored in PostgreSQL with their vector embeddings
2. **Query Processing**: User queries are converted to embeddings using OpenAI
3. **Similarity Search**: pgvector performs efficient cosine similarity search
4. **Context Retrieval**: Most similar documents are retrieved as context
5. **Response Generation**: OpenAI LLM generates responses using retrieved context


## Development

To run in development mode with auto-reload:
```bash
docker-compose up --build
```

The application will automatically restart when you make changes to the code.



## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**: Make sure your `.env` file contains a valid OpenAI API key
2. **Database Connection**: Ensure PostgreSQL container is healthy before the app starts
3. **Port Conflicts**: Change ports in `docker-compose.yml` if 5432 or 8000 are in use

### Logs
```bash
docker-compose logs rag-app
docker-compose logs postgres
```
