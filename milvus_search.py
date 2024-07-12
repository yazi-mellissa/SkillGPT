from typing import Any, List, Optional
from pymilvus import MilvusClient, model
import pandas as pd
import logging


collection = "skills"
skills_csv_file_path = "esco_embeddings/ESCO_khaled_version.csv"


class MilvusMemory:
    def __init__(self, milvus_path, wipe_milvus_on_start=False):
        self.client = MilvusClient(milvus_path)
        self.embedding_fn = model.dense.SentenceTransformerEmbeddingFunction(
            model_name='All-MPNet-Base-V2',
            device='cpu'
        )

        if wipe_milvus_on_start:
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
        try:
            skills_df = pd.read_csv(skills_csv_file_path)
            docs = skills_df['text'].tolist()
            expertises = skills_df['expertise'].tolist()

            vectors = self.embedding_fn.encode_documents(docs)

            data = [
                {"id": i, "vector": vectors[i], "text": docs[i], "expertise": expertises[i]}
                for i in range(len(docs))
            ]

            res = self.client.insert(
                collection_name=collection,
                data=data,
            )
            logging.info(f"Skills embeddings initialized: {res}")
        except Exception as e:
            logging.error(f"Error initializing skills embeddings: {e}")

    """
    def _setup_collection(self):
        if self.client.has_collection(collection_name=collection):
            self.client.drop_collection(collection_name=collection)

        skills_df = pd.read_csv(skills_csv_file_path)
        docs = skills_df['text'].tolist()
        expertises = skills_df['expertise'].tolist()

        vectors = self.embedding_fn.encode_documents(docs)

        # Insert data in batches
        batch_size = 1000
        for i in range(0, len(docs), batch_size):
            batch_data = [
                {"id": i + j, "vector": vectors[i + j], "text": docs[i + j], "expertise": expertises[i + j]}
                for j in range(min(batch_size, len(docs) - i))
            ]
            self.client.insert(
                collection_name=collection,
                data=batch_data,
            )
    """


    def get_relevant(
        self,
        query_vectors: List[float],
        collection: str = collection,
        num_relevant: int = 5
    ) -> Optional[List[Any]]:
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