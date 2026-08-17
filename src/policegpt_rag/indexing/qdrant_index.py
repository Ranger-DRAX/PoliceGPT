"""
Qdrant Hybrid Vector Store with dense vector + sparse inverted index + metadata filtering.
"""

from typing import List, Dict, Any, Optional
from loguru import logger
import uuid


class QdrantHybridIndex:
    def __init__(
        self,
        collection_name: str = "policegpt_legal_corpus",
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        dense_dim: int = 1024,
    ):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.api_key = api_key
        self.dense_dim = dense_dim
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(
                host=self.host,
                port=self.port,
                api_key=self.api_key if self.api_key else None,
            )
            return self._client
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant at {self.host}:{self.port} - {e}")
            return None

    def create_collection(self, recreate: bool = False):
        client = self._get_client()
        if client is None:
            return

        from qdrant_client.http import models as rest

        collections = client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists and recreate:
            logger.info(f"Recreating Qdrant collection: {self.collection_name}")
            client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            logger.info(f"Creating Qdrant collection: {self.collection_name}")
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": rest.VectorParams(size=self.dense_dim, distance=rest.Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": rest.SparseVectorParams(index=rest.SparseIndexParams(on_disk=False))
                },
            )

    def upsert_records(self, records: List[Dict[str, Any]], batch_size: int = 64):
        client = self._get_client()
        if client is None:
            logger.warning("Qdrant client unavailable for upsert.")
            return

        from qdrant_client.http import models as rest

        points = []
        for r in records:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, r["chunk_id"]))
            sparse_dict = r.get("sparse_weights", {})

            # Format sparse vector
            sparse_indices = [int(k) for k in sparse_dict.keys()]
            sparse_values = [float(v) for v in sparse_dict.values()]

            payload = {k: v for k, v in r.items() if k not in ("dense_vector", "sparse_weights")}

            vector_data = {"dense": r["dense_vector"]}
            if sparse_indices:
                vector_data["sparse"] = rest.SparseVector(
                    indices=sparse_indices, values=sparse_values
                )

            point = rest.PointStruct(
                id=point_id,
                vector=vector_data,
                payload=payload,
            )
            points.append(point)

        for i in range(0, len(points), batch_size):
            client.upsert(
                collection_name=self.collection_name,
                points=points[i : i + batch_size],
                wait=True,
            )
        logger.info(f"Successfully upserted {len(points)} records into Qdrant '{self.collection_name}'.")

    def hybrid_search(
        self,
        query_dense: List[float],
        query_sparse: Dict[int, float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        client = self._get_client()
        if client is None:
            return []

        from qdrant_client.http import models as rest

        # Construct payload filter if provided
        q_filter = None
        if filter_dict:
            must_conditions = []
            for k, v in filter_dict.items():
                must_conditions.append(
                    rest.FieldCondition(key=k, match=rest.MatchValue(value=v))
                )
            q_filter = rest.Filter(must=must_conditions)

        # Execute hybrid search with dense vectors
        search_results = client.search(
            collection_name=self.collection_name,
            query_vector=("dense", query_dense),
            query_filter=q_filter,
            limit=top_k,
            with_payload=True,
        )

        results = []
        for hit in search_results:
            item = dict(hit.payload or {})
            item["score"] = float(hit.score)
            item["chunk_id"] = item.get("chunk_id", str(hit.id))
            results.append(item)

        return results
