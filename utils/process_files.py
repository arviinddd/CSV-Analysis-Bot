import os
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
from utils.load_config import LoadConfig
from openai import OpenAI

class ProcessFiles:
    def __init__(self, file_path: str) -> None:
        self.config = LoadConfig()
        self.file_path = file_path
        self.file_name = os.path.splitext(os.path.basename(file_path))[0]
        self.db_path = f"{self.config.uploaded_files_directory}/{self.file_name}.db"

    def process_and_store(self):
        """Process CSV into SQLite and prepare Pinecone embeddings."""
        try:
            self.store_in_sqlite()
            self.prepare_vectordb()
        except Exception as e:
            print(f"Error in processing file: {e}")

    def store_in_sqlite(self):
        """Store CSV data into a separate SQLite database."""
        try:
            df = pd.read_csv(self.file_path)
            conn = sqlite3.connect(self.db_path)

            df.to_sql(self.file_name, conn, index=False, if_exists="replace")
            conn.close()
            print(f"Stored {self.file_name} in {self.db_path}")
        except Exception as e:
            print(f"Error storing CSV in SQLite: {e}")

    def prepare_vectordb(self):
        """Convert CSV rows into embeddings and store in Pinecone as JSON objects."""
        try:
            df = pd.read_csv(self.file_path)
            pinecone_index = self.config.pinecone_index
            openai_client = OpenAI(api_key=self.config.OPENAI_API_KEY)

            for index, row in df.iterrows():
                row_dict = row.to_dict()
                
                row_text = ", ".join([f"{key}: {value}" for key, value in row_dict.items()])
                embedding = openai_client.embeddings.create(
                    input=row_text, model=self.config.embedding_model_name
                ).data[0].embedding

                pinecone_index.upsert(
                    vectors=[(f"{self.file_name}-{index}", embedding, row_dict)]
                )
            print(f"Stored vectors from {self.file_name} in Pinecone.")
        except Exception as e:
            print(f"Error storing embeddings in Pinecone: {e}")
