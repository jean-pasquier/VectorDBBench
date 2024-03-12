from pydantic import BaseModel, SecretStr
from ..api import DBConfig, DBCaseConfig, MetricType, IndexType

POSTGRE_URL_PLACEHOLDER = "postgresql://%s:%s@%s/%s"


class PgVectorConfig(DBConfig):
    user_name: SecretStr = "postgres"
    password: SecretStr
    host: str = "localhost"
    port: int = 5432
    db_name: str

    def to_dict(self) -> dict:
        user_str = self.user_name.get_secret_value()
        pwd_str = self.password.get_secret_value()
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.db_name,
            "user": user_str,
            "password": pwd_str
        }


class PgVectorIndexConfig(BaseModel):
    index: IndexType
    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "vector_l2_ops"
        elif self.metric_type == MetricType.IP:
            return "vector_ip_ops"
        return "vector_cosine_ops"

    def parse_metric_fun_op(self) -> str:
        if self.metric_type == MetricType.L2:
            return "<->"
        elif self.metric_type == MetricType.IP:
            return "<#>"
        return "<=>"

    def parse_metric_fun_str(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2_distance"
        elif self.metric_type == MetricType.IP:
            return "max_inner_product"
        return "cosine_distance"


class HNSWConfig(PgVectorIndexConfig, DBCaseConfig):
    M: int
    efConstruction: int
    ef: int | None = None
    index: IndexType = IndexType.HNSW

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": "hnsw",
            "params": {"m": self.M, "ef_construction": self.efConstruction},
            # TODO parametrize these params
            "config": {"maintenance_work_mem": "'4GB'", "max_parallel_maintenance_workers": "4"}
        }

    def search_param(self) -> dict:
        return {
            "metric_fun_op": self.parse_metric_fun_op(),
            "metric_type": self.parse_metric(),
            "params": {"ef_search": self.ef},
        }


class IVFFlatConfig(PgVectorIndexConfig, DBCaseConfig):
    lists: int
    probes: int | None = None
    index: IndexType = IndexType.IVFFlat

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": "ivfflat",
            "params": {"lists": self.lists},
        }

    def search_param(self) -> dict:
        return {
            "metric_fun_op": self.parse_metric_fun_op(),
            "metric_type": self.parse_metric(),
            "params": {"probes": self.probes},
        }


_pgvector_case_config = {
    IndexType.HNSW: HNSWConfig,
    IndexType.IVFFlat: IVFFlatConfig
}
