from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import pandas as pd

from src.data_collection.formatters.base import DataFormatter
from src.data_collection.types import DataCollectionResult, RawPayload


class MappingFormatter(DataFormatter):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.column_map = dict(cfg.get("column_map", {}) or {})
        self.passthrough_fields = list(cfg.get("passthrough_fields", []) or [])
        self.sample_id_column = str(cfg.get("sample_id_column", "sample_id"))
        self.missing_value = cfg.get("missing_value", None)

    def _build_rows(
        self,
        records: List[Dict[str, Any]],
        column_map: Dict[str, str],
        passthrough_fields: List[str],
        sample_id_column: str,
        missing_value: Any,
        logger,
    ) -> pd.DataFrame:
        if not column_map:
            logger.warning("MappingFormatter column_map is empty; using raw keys as output columns.")
            df = pd.DataFrame(records)
        else:
            missing = Counter()
            rows = []
            for record in records:
                row = {}
                for out_col, raw_key in column_map.items():
                    if raw_key in record:
                        row[out_col] = record.get(raw_key)
                    else:
                        row[out_col] = missing_value
                        missing[raw_key] += 1
                for field in passthrough_fields:
                    if field in record:
                        row[field] = record.get(field)
                rows.append(row)
            df = pd.DataFrame(rows)
            if missing:
                missing_keys = ", ".join(sorted(missing.keys()))
                logger.warning("Missing raw fields for mapping: %s", missing_keys)

        if sample_id_column not in df.columns:
            df[sample_id_column] = [f"sample_{i:06d}" for i in range(len(df))]
        else:
            missing_mask = df[sample_id_column].isna() | (df[sample_id_column].astype(str).str.strip() == "")
            if missing_mask.any():
                fill_ids = [f"sample_{i:06d}" for i in range(len(df))]
                df.loc[missing_mask, sample_id_column] = [fill_ids[i] for i in df.index[missing_mask]]
        return df

    def format(self, payload: RawPayload, cfg: Dict[str, Any], logger) -> DataCollectionResult:
        column_map = dict(cfg.get("column_map", self.column_map) or {})
        passthrough_fields = list(cfg.get("passthrough_fields", self.passthrough_fields) or [])
        sample_id_column = str(cfg.get("sample_id_column", self.sample_id_column))
        missing_value = cfg.get("missing_value", self.missing_value)

        df = self._build_rows(
            payload.records,
            column_map,
            passthrough_fields,
            sample_id_column,
            missing_value,
            logger,
        )
        return DataCollectionResult(table=df, sdf_records=payload.sdf_records, metadata=payload.metadata)
