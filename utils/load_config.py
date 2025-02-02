import os
import yaml
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

class LoadConfig:
    def __init__(self) -> None:
        
        with open("configs/app_config.yml") as cfg:
            app_config = yaml.safe_load(cfg)

        self.uploaded_files_directory = app_config["directories"]["uploaded_files_directory"]
        self.embedding_model_name = app_config["llm_config"]["embedding_model_name"]
        self.pinecone_index_name = app_config["pinecone_config"]["index_name"]
        self.top_k = app_config["pinecone_config"]["top_k"]

        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not self.OPENAI_API_KEY:
            raise ValueError("OpenAI API Key is missing! Please check your .env file.")

        self.init_pinecone()

    def init_pinecone(self):
        """Initialize Pinecone client and create/connect to the index."""
        
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("Pinecone API Key is missing! Please check your .env file.")

        self.pinecone_client = Pinecone(api_key=api_key)

        if self.pinecone_index_name not in self.pinecone_client.list_indexes().names():
            print(f"Creating Pinecone index: {self.pinecone_index_name}")
            self.pinecone_client.create_index(
                name=self.pinecone_index_name,
                dimension=1536,  
                metric="cosine",  
                spec=ServerlessSpec(
                    cloud="aws",  
                    region="us-east-1"  
                )
            )
        else:
            print(f"Pinecone index '{self.pinecone_index_name}' already exists.")

        self.pinecone_index = self.pinecone_client.Index(self.pinecone_index_name)
        print(f"Connected to Pinecone index: {self.pinecone_index_name}")
