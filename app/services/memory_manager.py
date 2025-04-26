import logging
from pymilvus import connections, Collection, utility

logger = logging.getLogger(__name__)

class ZillizMilvusManager:
    """
    Manages Milvus deployment using Zilliz Cloud
    """
    
    def __init__(self, uri, token, db_name="default"):
        """
        Initialize the Zilliz Cloud manager
        
        Args:
            uri (str): Zilliz Cloud endpoint URI
            token (str): Zilliz Cloud API token
            db_name (str): Database name to use
        """
        self.uri = uri
        self.token = token
        self.db_name = db_name
        self.is_connected = False
        logger.info(f"ZillizMilvusManager initialized with URI: {uri}, database: {db_name}")
        
    def connect(self):
        """Connect to Zilliz Cloud"""
        try:
            connections.connect(
                alias="default", 
                uri=self.uri,
                token=self.token,
                db_name=self.db_name
            )
            self.is_connected = True
            logger.info(f"Successfully connected to Zilliz Cloud at {self.uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Zilliz Cloud: {e}")
            self.is_connected = False
            return False
            
    def disconnect(self):
        """Disconnect from Zilliz Cloud"""
        try:
            connections.disconnect("default")
            self.is_connected = False
            logger.info("Disconnected from Zilliz Cloud")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from Zilliz Cloud: {e}")
            return False
            
    def list_collections(self):
        """List all collections in the database"""
        try:
            if not self.is_connected:
                self.connect()
            return utility.list_collections()
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
            
    def has_collection(self, collection_name):
        """Check if a collection exists"""
        try:
            if not self.is_connected:
                self.connect()
            return utility.has_collection(collection_name)
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False
            
    def create_collection(self, collection_name, dimension):
        """Create a new collection for vector storage"""
        from pymilvus import CollectionSchema, FieldSchema, DataType
        
        try:
            if not self.is_connected:
                self.connect()
                
            if utility.has_collection(collection_name):
                logger.info(f"Collection {collection_name} already exists")
                return True
                
            # Define fields for the collection
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="expertise", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension)
            ]
            
            # Create collection schema
            schema = CollectionSchema(fields=fields, description=f"Collection for {collection_name}")
            
            # Create collection
            collection = Collection(name=collection_name, schema=schema)
            
            # Create an IVF_FLAT index for vector field
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index(field_name="vector", index_params=index_params)
            
            logger.info(f"Successfully created collection {collection_name} with dimension {dimension}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False
    
    def drop_collection(self, collection_name):
        """Drop a collection"""
        try:
            if not self.is_connected:
                self.connect()
                
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                logger.info(f"Dropped collection {collection_name}")
                return True
            else:
                logger.info(f"Collection {collection_name} does not exist")
                return False
                
        except Exception as e:
            logger.error(f"Failed to drop collection: {e}")
            return False