from typing import Any, List, Optional
import pandas as pd
import logging
from pymilvus import MilvusClient, Collection
from pymilvus import model
from fastapi import Depends
from app.core.config import get_settings

logger = logging.getLogger(__name__)

class MilvusMemoryService:
    """
    A service class for managing skill embeddings using Milvus vector database on Zilliz Cloud.
    """
    
    def __init__(self, settings = Depends(get_settings)):
        """
        Initializes a new instance of MilvusMemoryService with Zilliz Cloud.
        
        Args:
            settings: Application settings from dependency injection
        """
        self.uri = settings.ZILLIZ_URI
        self.token = settings.ZILLIZ_TOKEN
        self.client = MilvusClient(uri=self.uri, token=self.token)
        
        self.api_key = settings.OPEN_API_KEY
        self.embedding_fn = model.dense.OpenAIEmbeddingFunction(
            model_name='text-embedding-ada-002',
            api_key=self.api_key,
        )
        
        logger.info(f"MilvusMemoryService instance created with URI: {self.uri}")
        
        if settings.WIPE_MILVUS_ON_START:
            self._setup_collection()

    def _setup_collection(self) -> None:
        """
        Sets up the collection in Milvus, dropping the existing collection if it exists and reinitializes it with new data.
        """
        collection = "skills"
        try:
            if self.client.has_collection(collection_name=collection):
                self.client.drop_collection(collection_name=collection)
            
            self.client.create_collection(
                collection_name=collection,
                dimension=self.embedding_fn.dim,
            )
            logger.info(f"Collection '{collection}' created in Milvus")
            
        except Exception as e:
            logger.error(f"Error initializing Milvus collection: {e}")
            raise

    def init_skills_embeddings(self, skills_csv_file_path) -> None:
        """
        Initializes skills embeddings by reading data from a CSV file and inserting it into the Milvus collection.
        
        Args:
            skills_csv_file_path (str): Path to the CSV file containing skills data
        """
        collection = "skills"
        
        try:
            skills_df = pd.read_csv(skills_csv_file_path)
            docs = skills_df['text'].tolist()
            expertises = skills_df['expertise'].tolist()

            vectors = self.embedding_fn.encode_documents(docs)

            data = [
                {"id": i, "vector": vectors[i], "text": docs[i], "expertise": expertises[i]}
                for i in range(len(docs))
            ]

            self.client.insert(
                collection_name=collection,
                data=data,
            )
            logger.info(f"Inserted {len(data)} skills into Milvus collection")
            
        except Exception as e:
            logger.error(f"Error initializing skills embeddings: {e}")
            raise

    def encode_query(self, query: str) -> List[float]:
        """
        Encodes a query string into a vector embedding.
        
        Args:
            query (str): The query string to encode
            
        Returns:
            List[float]: The encoded vector
        """
        return self.embedding_fn.encode_queries([query])

    def get_relevant(
        self,
        query_vectors: List[float],
        collection: str = "skills",
        num_relevant: int = 5
    ) -> Optional[List[Any]]:
        """
        Retrieves relevant documents based on provided query vectors.
        
        Args:
            query_vectors (List[float]): The embedding vector of the query.
            collection (str, optional): The collection to search within. Defaults to 'skills'.
            num_relevant (int, optional): Number of relevant results to retrieve. Defaults to 5.
        
        Returns:
            Optional[List[Any]]: List of relevant documents, or None if an error occurs.
        """
        try:
            results = self.client.search(
                collection_name=collection,
                data=query_vectors,
                limit=num_relevant,
                output_fields=["expertise", "text"],
            )
            logger.info(f"Retrieved {len(results)} relevant documents from Milvus")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving relevant documents: {e}")
            raise