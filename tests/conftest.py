"""
Shared fixtures: a tiny hand-checkable Olist database and a deterministic fake embedder,
so the full query path (guard -> planner -> DuckDB -> PII vault) runs in CI without models.

Data (every number below is used by assertions):
  orders      O1 (C1, SP)  items: S1 10.00 + S2 20.00   payments: 15.00 + 15.00   reviews: R1 broken, R5 broken
              O2 (C2, RJ)  items: S1 100.00             payments: 100.00          reviews: R2 late
              O3 (C3, RJ)  items: S2 50.00              payments: 50.00           reviews: R3 great
              O4 (C1, SP)  items: S2 5.00               payments: 5.00            reviews: R4 broken
"""

import zlib

import numpy as np
import pytest

from dualquery.engine.executor import HybridExecutor
from dualquery.ingestion.olist_loader import OlistLoader

VOCAB = ["broken", "late", "great", "missing"]


class FakeEmbedder:
    """Bag-of-keywords embedding: texts sharing a keyword are close, others are orthogonal."""
    dimension = 768

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dimension, dtype=np.float32)
        hits = [i for i, w in enumerate(VOCAB) if w in text.lower()]
        for i in hits:
            v[i] = 1.0
        if not hits:  # unique direction per unrelated text, stable across runs
            v[10 + zlib.crc32(text.encode()) % 700] = 1.0
        return (v / np.linalg.norm(v)).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_documents(self, texts: list[str], batch_size: int = 128) -> list[list[float]]:
        return [self._vec(t) for t in texts]


@pytest.fixture(scope="session")
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture(scope="session")
def tiny_db(tmp_path_factory, fake_embedder) -> str:
    db_path = tmp_path_factory.mktemp("olist") / "tiny.duckdb"
    loader = OlistLoader(db_path=db_path, embedding_provider=fake_embedder)
    conn = loader.connect()
    try:
        loader.initialize_schema(conn)
        conn.execute("""
            INSERT INTO customers VALUES
                ('C1', 'U1', '01000', 'sao paulo', 'SP'),
                ('C2', 'U2', '20000', 'rio de janeiro', 'RJ'),
                ('C3', 'U3', '20001', 'rio de janeiro', 'RJ');
            INSERT INTO orders (order_id, customer_id, order_status, order_purchase_timestamp,
                                order_delivered_customer_date, order_estimated_delivery_date) VALUES
                ('O1', 'C1', 'delivered', '2018-01-01', '2018-01-10', '2018-01-08'),
                ('O2', 'C2', 'delivered', '2018-01-02', '2018-01-05', '2018-01-09'),
                ('O3', 'C3', 'canceled',  '2018-01-03', NULL, '2018-01-10'),
                ('O4', 'C1', 'delivered', '2018-01-04', '2018-01-06', '2018-01-07');
            INSERT INTO products VALUES ('P1', 'utilidades', 'housewares', 500), ('P2', 'brinquedos', 'toys', 300);
            INSERT INTO order_items VALUES
                ('O1', 1, 'P1', 'S1', 10.00, 1.00),
                ('O1', 2, 'P2', 'S2', 20.00, 2.00),
                ('O2', 1, 'P1', 'S1', 100.00, 3.00),
                ('O3', 1, 'P2', 'S2', 50.00, 4.00),
                ('O4', 1, 'P2', 'S2', 5.00, 5.00);
            INSERT INTO order_payments VALUES
                ('O1', 1, 'credit_card', 1, 15.00),
                ('O1', 2, 'voucher', 1, 15.00),
                ('O2', 1, 'boleto', 1, 100.00),
                ('O3', 1, 'credit_card', 3, 50.00),
                ('O4', 1, 'credit_card', 1, 5.00);
        """)
        reviews = [
            ("R1", "O1", 1, "box arrived broken, call me at joao@mail.com"),
            ("R2", "O2", 2, "late delivery, very slow"),
            ("R3", "O3", 5, "great product"),
            ("R4", "O4", 1, "item broken on arrival"),
            ("R5", "O1", 1, "second item also broken"),
        ]
        for rid, oid, score, text in reviews:
            conn.execute(
                "INSERT INTO order_reviews VALUES (?, ?, ?, NULL, ?, '2018-02-01', ?)",
                [rid, oid, score, text, fake_embedder.embed_query(text)],
            )
        loader.build_indexes(conn)
    finally:
        conn.close()
    return str(db_path)


@pytest.fixture
def executor(tiny_db, fake_embedder) -> HybridExecutor:
    return HybridExecutor(db_path=tiny_db, embedder=fake_embedder)
