import os
import pandas as pd
from utils.load_config import LoadConfig

class PrepareVectorDB:
    def __init__(self, file_path: str) -> None:
        self.config = LoadConfig()
        self.file_path = file_path
        self.file_name = os.path.splitext(os.path.basename(file_path))[0]

    def process_and_store_vectors(self):
        """Convert CSV rows into vector embeddings and store in Pinecone"""
        df = pd.read_csv(self.file_path)
        pinecone_index = self.config.pinecone_index

        for index, row in df.iterrows():
            row_text = ", ".join([f"{col}: {row[col]}" for col in df.columns])
            embedding = self.config.azure_openai_client.embeddings.create(
                input=row_text, model=self.config.embedding_model_name
            ).data[0].embedding

            pinecone_index.upsert(vectors=[(f"{self.file_name}-{index}", embedding, {"source": self.file_name})])
        print(f"Stored vectors from {self.file_name} in Pinecone.")