from typing import Any, List, Optional
from pymilvus import MilvusClient, model
import pandas as pd

collection = "skills"
#dimension = 30522
skills_csv_file_path = "esco_embeddings/ESCO_khaled_version.csv"


class MilvusMemory:
    def __init__(self, milvus_path, api_key ,wipe_milvus_on_start=False):
        self.client = MilvusClient(milvus_path)
        self.embedding_fn = model.dense.SentenceTransformerEmbeddingFunction(
            model_name='all-MiniLM-L6-v2',
            device='cpu'
        )


        if wipe_milvus_on_start:
            if self.client.has_collection(collection_name=collection):
                self.client.drop_collection(collection_name=collection)

            self.client.create_collection(
                collection_name=collection,
                dimension=self.embedding_fn.dim, 
            )
            self.init_skills_embeddings()
        
    
    def init_skills_embeddings(self) -> None:
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
        print(res)


    def get_relevant(
        self,
        query_vectors: List[float],
        collection: str,
        num_relevant: int = 5
    ) -> Optional[List[Any]]:
        return self.client.search(
            collection_name=collection,
            data=query_vectors,
            limit=num_relevant,

            output_fields=["id","expertise"],
        )