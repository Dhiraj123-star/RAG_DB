-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table for storing documents and embeddings
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding VECTOR(1536), -- OpenAI embedding dimension
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Insert some sample data
INSERT INTO documents (content, metadata) VALUES 
('Python is a high-level programming language known for its simplicity and readability.', '{"category": "programming", "topic": "python"}'),
('Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.', '{"category": "ai", "topic": "machine_learning"}'),
('Docker is a containerization platform that allows you to package applications and their dependencies.', '{"category": "devops", "topic": "docker"}'),
('PostgreSQL is a powerful, open-source relational database management system.', '{"category": "database", "topic": "postgresql"}'),
('LangChain is a framework for developing applications powered by language models.', '{"category": "ai", "topic": "langchain"}');