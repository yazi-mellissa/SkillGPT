from gptcache import Cache
from gptcache.adapter.openai import OpenAI as GPTCacheOpenAI
from gptcache.embedding import OpenAIEmbedding
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.processor.pre import get_prompt
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
import logging

logger = logging.getLogger(__name__)

class GPTCacheManager:
    """
    Manages GPTCache integration with OpenAI to reduce API calls and improve performance.
    """
    
    def __init__(self, openai_api_key, zilliz_uri=None, zilliz_token=None):
        """
        Initialize GPTCache with Zilliz Cloud as the vector store.
        
        Args:
            openai_api_key (str): OpenAI API key for embeddings
            zilliz_uri (str): URI for Zilliz Cloud instance
            zilliz_token (str): Token for Zilliz Cloud authentication
        """
        self.openai_api_key = openai_api_key
        self.zilliz_uri = zilliz_uri
        self.zilliz_token = zilliz_token
        self.cache = None
        self.setup_cache()
        
    def setup_cache(self):
        """Set up GPTCache with appropriate storage backend"""
        embedding = OpenAIEmbedding(api_key=self.openai_api_key, model="text-embedding-ada-002")
        
        if self.zilliz_uri and self.zilliz_token:
            # Connect to Zilliz Cloud for vector storage
            data_manager = get_data_manager(
                CacheBase("sqlite"),  # Metadata storage
                VectorBase("milvus", 
                           uri=self.zilliz_uri,
                           token=self.zilliz_token,
                           collection_name="gptcache",
                           dimension=1536)  # Vector storage
            )
            logger.info("GPTCache initialized with Zilliz Cloud backend")
        else:
            # Fallback to SQLite and FAISS
            data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("faiss", dimension=1536))
            logger.info("GPTCache initialized with local FAISS backend")
        
        self.cache = Cache()
        self.cache.init(
            pre_embedding_func=get_prompt,
            embedding_func=embedding.to_embeddings,
            data_manager=data_manager,
            similarity_evaluation=SearchDistanceEvaluation(),
            cache_policy_param={"capacity": 1000}
        )
        
    def get_cached_client(self, original_client):
        """
        Return a GPTCache-enabled OpenAI client that will use cache when possible.
        
        Args:
            original_client: The original OpenAI client
            
        Returns:
            GPTCacheOpenAI: A cache-enabled OpenAI client
        """
        return GPTCacheOpenAI(cache_obj=self.cache, client=original_client)