import sqlite3
from utils.load_config import LoadConfig
from sqlalchemy import create_engine
import openai

class CSVAnalysisBot:
    def __init__(self):
        self.config = LoadConfig()
        self.openai_client = openai

    def decide_query_type(self, question):
        """Decide whether to use SQL or semantic search."""
        sql_keywords = ["list", "count", "show", "select", "retrieve", "order"]
        for keyword in sql_keywords:
            if keyword in question.lower():
                return "sql"
        return "semantic"

    def respond(self, message: str, file_name: str) -> str:
        query_type = self.decide_query_type(message)

        if query_type == "sql":
            return self.handle_sql_query(message, file_name)
        elif query_type == "semantic":
            return self.handle_semantic_search(message)
        else:
            return "Unable to process the question."

    def handle_sql_query(self, question: str, file_name: str) -> str:
        """Generate and execute an SQL query based on the question."""
        sql_query = self.generate_sql_query(question, file_name)
        if not sql_query:
            return "Sorry, I couldn't generate a valid SQL query. Please try again."

        db_path = f"{self.config.uploaded_files_directory}/{file_name}.db"
        query_result = self.execute_sql_query(sql_query, db_path)
        if not query_result:
            return "No relevant data found in the database."

        return self.generate_concise_response(query_result)

    def handle_semantic_search(self, message: str) -> str:
        """Perform semantic search using Pinecone and refine with OpenAI LLM."""
        try:
            response = self.openai_client.embeddings.create(
                input=[message],  
                model=self.config.embedding_model_name
            )
            query_embedding = response.data[0].embedding

            search_results = self.config.pinecone_index.query(
                vector=query_embedding,
                top_k=self.config.top_k,
                include_metadata=True
            )

            matches = search_results.get("matches", [])
            if not matches:
                return "No relevant results found in the database."

            extracted_records = []
            for match in matches:
                metadata = match.get("metadata", {})

                if not metadata:
                    continue  

            
                formatted_data = ", ".join([f"{key}: {value}" for key, value in metadata.items()])
                extracted_records.append(f"Record: {formatted_data}")

            retrieved_data = "\n".join(extracted_records)

            refined_response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an AI assistant summarizing extracted data into a natural response."},
                    {"role": "user", "content": f"I asked: '{message}'\nHere is the extracted data:\n{retrieved_data}\nRephrase this into a natural answer."}
                ],
                max_tokens=100,
                temperature=0.7
            )

            return refined_response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error during semantic search: {e}")
            return "Error performing semantic search. Please try again."



    def generate_sql_query(self, question: str, table_name: str) -> str:
        """Generate SQL query from natural language input."""
        prompt = f"""
        Convert the following question into an SQL query:
        Question: {question}
        Table: {table_name}
        SQL Query:
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert SQL generator."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Error generating SQL query: {e}")
            return None

    def execute_sql_query(self, sql_query: str, db_path: str) -> list:
        """Execute the SQL query on the SQLite database."""
        try:
            engine = create_engine(f"sqlite:///{db_path}")
            with engine.connect() as conn:
                result = conn.execute(sql_query).fetchall()
                return result
        except Exception as e:
            print(f"Error executing SQL query: {e}")
            return None

    def generate_concise_response(self, query_result: list) -> str:
        """Generate a short, precise response from the query result."""
        prompt = f"""
        The database query returned the following result: {query_result}
        Generate a short, precise answer based only on this result. Avoid unnecessary words.
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert summarizer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.5
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Error generating concise response: {e}")
            return "Sorry, I couldn't generate a meaningful response."
