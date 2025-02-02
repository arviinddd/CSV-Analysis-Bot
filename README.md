# CSV Analysis Bot

## Overview
CSV Analysis Bot is an AI-powered system designed to process, analyze, and query CSV files efficiently. It leverages **SQL databases** and **vector databases (Pinecone)** for structured and semantic search queries, enabling users to retrieve insights from uploaded CSV data using natural language.

---

## Key Features
- Automatic CSV Processing: Stores structured data in **SQLite** and converts rows into vector embeddings.
- Intelligent Query Handling: Determines whether to use **SQL queries** or **semantic search** dynamically.
- SQL Query Generation: Uses **OpenAI GPT-4** to generate and execute SQL queries from natural language prompts.
- Semantic Search with Vector DB: Stores and retrieves contextual embeddings using **Pinecone**.
- Natural Language Responses: Uses **OpenAI LLM** to refine extracted search results into human-readable responses.
- Flexible Metadata Handling: Adapts to different CSV structures dynamically.
- User-friendly Interface: Built with **Gradio** for seamless interaction.

---

## Project Workflow
1. Upload a CSV file.
2. Data Processing:
   - Structured data is stored in **SQLite**.
   - Unstructured data is converted into **vector embeddings**.
3. Query Handling:
   - If the query is SQL-like ("list", "count", "retrieve", etc.), it executes an **SQL query**.
   - Otherwise, it performs **semantic search** using vector similarity in Pinecone.
4. Result Refinement: The extracted data is formatted into a human-readable response using **OpenAI LLM**.

---

## Technologies Used

### Backend
- **Python** - Core language for processing and logic.

### Data Processing & Storage
- **Pandas** - For reading and manipulating CSV files.
- **SQLite** - Stores structured tabular data.
- **SQLAlchemy** - ORM for handling database queries.

### Machine Learning & AI
- **OpenAI GPT-4** - Generates SQL queries and refines search results.
- **OpenAI Embeddings API** - Converts text data into vector representations.
- **Pinecone** - Vector database for semantic search.

### User Interface
- **Gradio** - Interactive UI for uploading CSV files and querying.

