from __future__ import annotations

from src.featuresets.base import GraphFeatureSet, SmilesFeatureSet, TabularFeatureSet
from src.featuresets.registry import register_featureset


_TABULAR_FEATURESETS = [
    "fp_morgan",
    "fp_morgan_desc",
    "fp_morgan_quick",
    "fp_morgan_fixture",
    "pretrained_embedding",
    "pretrained_embedding_stub",
]

_GRAPH_FEATURESETS = [
    "graph",
    "gnn_graph",
    "gnn_graph_quick",
    "gnn_radius_graph",
    "gnn_radius_graph_quick",
    "gnn2d",
]

_SMILES_FEATURESETS = [
    "smiles_tokenizer",
    "smiles_tokenizer_qm9",
]

for name in _TABULAR_FEATURESETS:
    register_featureset(name, TabularFeatureSet)

for name in _GRAPH_FEATURESETS:
    register_featureset(name, GraphFeatureSet)

for name in _SMILES_FEATURESETS:
    register_featureset(name, SmilesFeatureSet)
