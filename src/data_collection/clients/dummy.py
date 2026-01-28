from __future__ import annotations

from typing import Any, Dict, List

from src.data_collection.clients.base import DataClient
from src.data_collection.types import DataCollectionQuery, RawPayload


class DummyClient(DataClient):
    def fetch(self, query: DataCollectionQuery) -> RawPayload:
        records_cfg = self.cfg.get("records")
        if isinstance(records_cfg, list) and records_cfg:
            records = [dict(item) for item in records_cfg]
        else:
            records = [
                {
                    "cas": "64-17-5",
                    "formula": "C2H6O",
                    "tc_k": 514.0,
                    "pc_pa": 6137000.0,
                    "tb_k": 351.5,
                    "source": "dummy",
                },
                {
                    "cas": "67-64-1",
                    "formula": "C3H6O",
                    "tc_k": 508.0,
                    "pc_pa": 4700000.0,
                    "tb_k": 329.4,
                    "source": "dummy",
                },
            ]

        if query.identifiers:
            wanted = set(str(x) for x in query.identifiers)
            records = [r for r in records if str(r.get("cas")) in wanted]

        if query.limit is not None:
            records = records[: int(query.limit)]

        metadata = {"dummy_count": len(records)}
        return RawPayload(records=records, sdf_records={}, metadata=metadata)
