from typing import Any, List, Optional
from pymilvus import MilvusClient, model
import pandas as pd
import logging

collection = "skills"
skills_csv_file_path = "esco_embeddings/ESCO_khaled_version.csv"

class MilvusMemory:
    """
    A class for managing skill embeddings using Milvus vector database.
    """
    
    def __init__(self, milvus_path: str, wipe_milvus_on_start: bool = False):
        """
        Initializes a new instance of MilvusMemory.
        
        Args:
            milvus_path (str): Path to the Milvus server.
            wipe_milvus_on_start (bool, optional): Determines whether to wipe the existing Milvus collection on startup. Defaults to False.
        """
        self.client = MilvusClient(milvus_path)
        self.embedding_fn = model.dense.SentenceTransformerEmbeddingFunction(
            model_name='All-MPNet-Base-V2',
            device='cpu'
        )
        
        if wipe_milvus_on_start:
            self._setup_collection()

    def _setup_collection(self) -> None:
        """
        Sets up the collection in Milvus, dropping the existing collection if it exists and reinitializes it with new data.
        """
        try:
            if self.client.has_collection(collection_name=collection):
                self.client.drop_collection(collection_name=collection)
            
            self.client.create_collection(
                collection_name=collection,
                dimension=self.embedding_fn.dim,
            )
            self.init_skills_embeddings()
        except Exception as e:
            logging.error(f"Error initializing Milvus collection: {e}")

    def init_skills_embeddings(self) -> None:
        """
        Initializes skills embeddings by reading data from a CSV file and inserting it into the Milvus collection.
        """
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
            logging.info("Skills embeddings initialized.")
        except Exception as e:
            logging.error(f"Error initializing skills embeddings: {e}")

    def get_relevant(
        self,
        query_vectors: List[float],
        collection: str = collection,
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
                output_fields=["expertise"],
            )
            return results
        except Exception as e:
            logging.error(f"Error retrieving relevant documents: {e}")
            return None
