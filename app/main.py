import os
import asyncio
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncpg
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Simple RAG Application", version="1.0.0")

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag_user:rag_password@localhost:5432/rag_db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Initialize components
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

def vector_to_db_format(vector):
    """Convert vector to PostgreSQL format"""
    return '[' + ','.join(map(str, vector)) + ']'

# Custom retriever class for PostgreSQL with pgvector
class PostgreSQLRetriever:
    def __init__(self, database_url: str, embeddings_model, top_k: int = 3):
        self.database_url = database_url
        self.embeddings_model = embeddings_model
        self.top_k = top_k
        self.pool = None
    
    async def initialize(self):
        """Initialize the connection pool"""
        self.pool = await asyncpg.create_pool(self.database_url)
        await self._update_embeddings()
    
    async def _update_embeddings(self):
        """Generate embeddings for documents that don't have them"""
        async with self.pool.acquire() as conn:
            # Get documents without embeddings
            docs = await conn.fetch("SELECT id, content FROM documents WHERE embedding IS NULL")
            
            for doc in docs:
                try:
                    # Generate embedding
                    embedding = await asyncio.to_thread(
                        self.embeddings_model.embed_query, doc['content']
                    )
                    
                    # Convert to DB format
                    embedding_db = vector_to_db_format(embedding)

                    # Update document with embedding
                    await conn.execute(
                        "UPDATE documents SET embedding = $1::vector WHERE id = $2",
                        embedding_db, doc['id']
                    )
                    logger.info(f"Updated embedding for document {doc['id']}")
                except Exception as e:
                    logger.error(f"Error generating embedding for document {doc['id']}: {e}")
    
    async def retrieve(self, query: str) -> List[Document]:
        """Retrieve similar documents based on query"""
        try:
            # Generate query embedding
            query_embedding = await asyncio.to_thread(
                self.embeddings_model.embed_query, query
            )
            query_embedding_db = vector_to_db_format(query_embedding)
            
            async with self.pool.acquire() as conn:
                # Find similar documents using cosine similarity
                similar_docs = await conn.fetch("""
                    SELECT content, metadata, (1 - (embedding <=> $1::vector)) as similarity
                    FROM documents 
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                """, query_embedding_db, self.top_k)
                
                # Convert to LangChain Document format
                documents = []
                for doc in similar_docs:
                    metadata = {}
                    if doc['metadata']:
                        try:
                            metadata = json.loads(doc['metadata'])
                        except Exception:
                            metadata = {"raw_metadata": str(doc['metadata'])}
                    
                    metadata['similarity'] = doc['similarity']
                    documents.append(Document(
                        page_content=doc['content'],
                        metadata=metadata
                    ))
                
                return documents
                
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

# Initialize retriever
retriever = PostgreSQLRetriever(DATABASE_URL, embeddings)

# Custom RAG chain
class SimpleRAGChain:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = PromptTemplate(
            template="""You are a helpful assistant. Use the following context to answer the question. 
            If you cannot find the answer in the context, say so clearly.

            Context:
            {context}

            Question: {question}

            Answer: """,
            input_variables=["context", "question"]
        )
    
    async def run(self, query: str) -> str:
        try:
            # Retrieve relevant documents
            docs = await self.retriever.retrieve(query)
            
            if not docs:
                return "I couldn't find any relevant information in the database to answer your question."
            
            # Prepare context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate prompt
            prompt = self.prompt_template.format(context=context, question=query)
            
            # Get response from LLM
            response = await asyncio.to_thread(self.llm.predict, prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG chain: {e}")
            return f"An error occurred while processing your query: {str(e)}"

# Initialize RAG chain
rag_chain = SimpleRAGChain(retriever, llm)

# Request/Response models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_docs: Optional[List[dict]] = None

class AddDocumentRequest(BaseModel):
    content: str
    metadata: Optional[dict] = {}

# API endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize the retriever on startup"""
    await retriever.initialize()
    logger.info("RAG application started successfully!")

@app.get("/")
async def root():
    return {"message": "Simple RAG Application is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system"""
    try:
        # Get retrieved documents for transparency
        retrieved_docs = await retriever.retrieve(request.query)
        
        # Get answer from RAG chain
        answer = await rag_chain.run(request.query)
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            retrieved_docs=[
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity": doc.metadata.get("similarity", 0)
                }
                for doc in retrieved_docs
            ]
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-document")
async def add_document(request: AddDocumentRequest):
    """Add a new document to the database"""
    try:
        logger.info(f"Adding document with content: {request.content[:100]}...")
        
        async with retriever.pool.acquire() as conn:
            # Convert metadata to JSON string
            metadata_json = json.dumps(request.metadata) if request.metadata else '{}'
            
            # Insert document with JSON metadata
            doc_id = await conn.fetchval(
                "INSERT INTO documents (content, metadata) VALUES ($1, $2) RETURNING id",
                request.content, metadata_json
            )
            logger.info(f"Inserted document with ID: {doc_id}")
            
            # Generate and update embedding
            logger.info("Generating embedding...")
            embedding = await asyncio.to_thread(
                embeddings.embed_query, request.content
            )
            logger.info(f"Generated embedding with {len(embedding)} dimensions")
            
            # Convert to database format
            embedding_db = vector_to_db_format(embedding)
            
            # Update document with embedding
            await conn.execute(
                "UPDATE documents SET embedding = $1::vector WHERE id = $2",
                embedding_db, doc_id
            )
            logger.info(f"Updated document {doc_id} with embedding")
            
            return {"message": "Document added successfully", "id": doc_id}
            
    except Exception as e:
        logger.error(f"Error adding document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """List all documents in the database"""
    try:
        async with retriever.pool.acquire() as conn:
            docs = await conn.fetch("SELECT id, content, metadata, created_at FROM documents ORDER BY created_at DESC")
            return [
                {
                    "id": doc["id"],
                    "content": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                    "metadata": json.loads(doc["metadata"]) if doc["metadata"] else {},
                    "created_at": doc["created_at"]
                }
                for doc in docs
            ]
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debug/generate-embeddings")
async def generate_embeddings():
    """Manually trigger embedding generation for documents without embeddings"""
    try:
        await retriever._update_embeddings()
        
        # Check status after generation
        async with retriever.pool.acquire() as conn:
            docs_with_embeddings = await conn.fetchval("SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL")
            docs_without_embeddings = await conn.fetchval("SELECT COUNT(*) FROM documents WHERE embedding IS NULL")
            total_docs = await conn.fetchval("SELECT COUNT(*) FROM documents")

            sample_docs = await conn.fetch("""
                SELECT id, content, 
                       CASE WHEN embedding IS NOT NULL THEN 'YES' ELSE 'NO' END as has_embedding,
                       metadata
                FROM documents 
                ORDER BY id 
                LIMIT 5
            """)
        
        return {
            "message": "Embedding generation completed",
            "total_documents": total_docs,
            "documents_with_embeddings": docs_with_embeddings,
            "documents_without_embeddings": docs_without_embeddings,
            "sample_documents": [
                {
                    "id": doc["id"],
                    "content_preview": doc["content"][:100] + "..." if len(doc["content"]) > 100 else doc["content"],
                    "has_embedding": doc["has_embedding"],
                    "metadata": json.loads(doc["metadata"]) if doc["metadata"] else {}
                }
                for doc in sample_docs
            ]
        }
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
