import logging

from vectordb_bench.backend.clients import DBCaseConfig, IndexType
from vectordb_bench.backend.clients.pgvector.pgvector import PgVector
from vectordb_bench.backend.clients.pgvector.config import HNSWConfig, IVFFlatConfig


log = logging.getLogger(__name__)


class CitusHNSWConfig(HNSWConfig):
    distribution_col: str = None


class CitusIVFFlatConfig(IVFFlatConfig):
    distribution_col: str = None


class PgOnCitus(PgVector):
    """Differences from pgvector on postgres:
    1. SELECT CREATE_EXTENSION('vector');
    2. CREATE TABLE abc + create_distributed_table('abc', arguments...)

    TODO:
        1. Add random metadata in dataset loading to shard table on
        2. Add options on queries such as where metadata = X or metadata in(Y, Z), etc.
        3. Create hybrid search/RRF queries (requires TEXT + embeddings)

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_extension(self):
        self.cursor.execute("SELECT create_extension('vector');")

    def _create_table(self, dim: int):
        super()._create_table(dim)
        query = f"SELECT create_distributed_table('{self.table_name}', '{self.db_config.distribution_col}')"
        log.info(f"pgVector on Citus: running '{query}'")
        self.cursor.execute(query)
        self.conn.commit()


_pgvector_citus_case_config = {
    IndexType.HNSW: CitusHNSWConfig,
    IndexType.IVFFlat: CitusIVFFlatConfig
}