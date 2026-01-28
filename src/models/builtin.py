from __future__ import annotations

from typing import Any, Dict

from src.models.registry import ModelCapabilities, register_model


def _fp_factory(model_name: str):
    def _factory(model_cfg: Dict[str, Any], context: Dict[str, Any]):
        from src.fp.models import get_model

        params = model_cfg.get("params", {}) or {}
        return get_model(model_name, params)

    return _factory


def _require_context(context: Dict[str, Any], key: str) -> int:
    if key not in context:
        raise ValueError(f"Model context missing required key: {key}")
    return int(context[key])


def _context_int(context: Dict[str, Any], key: str, default: int = 0) -> int:
    value = context.get(key, default)
    if value is None:
        value = default
    return int(value)


def _gnn_config(model_cfg: Dict[str, Any]) -> tuple[int, int, float]:
    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    num_layers = int(model_cfg.get("num_layers", 4))
    dropout = float(model_cfg.get("dropout", 0.1))
    return hidden_dim, num_layers, dropout


def _gnn_gcn_factory(model_cfg: Dict[str, Any], context: Dict[str, Any]):
    from src.gnn.models import GCNRegressor

    in_dim = _require_context(context, "in_dim")
    global_dim = _context_int(context, "global_dim", 0)
    out_dim = _context_int(context, "out_dim", 1)
    hidden_dim, num_layers, dropout = _gnn_config(model_cfg)
    return GCNRegressor(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        global_dim=global_dim,
        out_dim=out_dim,
    )


def _gnn_gin_factory(model_cfg: Dict[str, Any], context: Dict[str, Any]):
    from src.gnn.models import GINRegressor

    in_dim = _require_context(context, "in_dim")
    edge_dim = _context_int(context, "edge_dim", 0)
    global_dim = _context_int(context, "global_dim", 0)
    out_dim = _context_int(context, "out_dim", 1)
    hidden_dim, num_layers, dropout = _gnn_config(model_cfg)
    return GINRegressor(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        global_dim=global_dim,
        edge_dim=edge_dim,
        out_dim=out_dim,
    )


def _gnn_mpnn_factory(model_cfg: Dict[str, Any], context: Dict[str, Any]):
    from src.gnn.models import MPNNRegressor

    in_dim = _require_context(context, "in_dim")
    edge_dim = _context_int(context, "edge_dim", 0)
    if edge_dim <= 0:
        raise ValueError("MPNN requires edge_attr with dim > 0; check featurizer.edge_features.")
    global_dim = _context_int(context, "global_dim", 0)
    out_dim = _context_int(context, "out_dim", 1)
    hidden_dim, num_layers, dropout = _gnn_config(model_cfg)
    edge_mlp_hidden_dim = int(model_cfg.get("edge_mlp_hidden_dim", 128))
    return MPNNRegressor(
        in_dim=in_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        global_dim=global_dim,
        edge_mlp_hidden_dim=edge_mlp_hidden_dim,
        out_dim=out_dim,
    )


def _gnn_schnet_factory(model_cfg: Dict[str, Any], context: Dict[str, Any]):
    from src.gnn.models import SchNetRegressor

    hidden_dim = int(model_cfg.get("hidden_dim", model_cfg.get("hidden_channels", 128)))
    num_filters = int(model_cfg.get("num_filters", hidden_dim))
    num_interactions = int(model_cfg.get("num_interactions", model_cfg.get("num_layers", 6)))
    num_gaussians = int(model_cfg.get("num_gaussians", model_cfg.get("num_kernels", 50)))
    cutoff = float(model_cfg.get("cutoff", model_cfg.get("radius_cutoff", 5.0)))
    readout = str(model_cfg.get("readout", "add"))
    return SchNetRegressor(
        hidden_dim=hidden_dim,
        num_filters=num_filters,
        num_interactions=num_interactions,
        num_gaussians=num_gaussians,
        cutoff=cutoff,
        readout=readout,
    )


def _gnn_dimenetpp_factory(model_cfg: Dict[str, Any], context: Dict[str, Any]):
    from src.gnn.models import DimeNetPPRegressor

    out_dim = _context_int(context, "out_dim", 1)
    if out_dim != 1:
        raise ValueError("DimeNet++ supports a single regression target.")
    hidden_dim = int(model_cfg.get("hidden_dim", model_cfg.get("hidden_channels", 128)))
    num_blocks = int(model_cfg.get("num_blocks", 4))
    num_bilinear = int(model_cfg.get("num_bilinear", 8))
    num_spherical = int(model_cfg.get("num_spherical", 7))
    num_radial = int(model_cfg.get("num_radial", 6))
    cutoff = float(model_cfg.get("cutoff", model_cfg.get("radius_cutoff", 5.0)))
    max_num_neighbors = int(model_cfg.get("max_num_neighbors", model_cfg.get("max_neighbors", 32)))
    envelope_exponent = int(model_cfg.get("envelope_exponent", 5))
    num_before_skip = int(model_cfg.get("num_before_skip", 1))
    num_after_skip = int(model_cfg.get("num_after_skip", 2))
    num_output_layers = int(model_cfg.get("num_output_layers", 3))
    int_emb_size = model_cfg.get("int_emb_size", model_cfg.get("interaction_emb_size", None))
    basis_emb_size = model_cfg.get("basis_emb_size", None)
    out_emb_channels = model_cfg.get("out_emb_channels", None)
    return DimeNetPPRegressor(
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_blocks=num_blocks,
        num_bilinear=num_bilinear,
        num_spherical=num_spherical,
        num_radial=num_radial,
        cutoff=cutoff,
        max_num_neighbors=max_num_neighbors,
        envelope_exponent=envelope_exponent,
        num_before_skip=num_before_skip,
        num_after_skip=num_after_skip,
        num_output_layers=num_output_layers,
        int_emb_size=None if int_emb_size is None else int(int_emb_size),
        basis_emb_size=None if basis_emb_size is None else int(basis_emb_size),
        out_emb_channels=None if out_emb_channels is None else int(out_emb_channels),
    )


def _gnn_painn_factory(model_cfg: Dict[str, Any], context: Dict[str, Any]):
    from src.gnn.models import PaiNNRegressor

    out_dim = _context_int(context, "out_dim", 1)
    if out_dim != 1:
        raise ValueError("PaiNN supports a single regression target.")
    in_dim = _context_int(context, "in_dim", 0)
    hidden_dim = int(model_cfg.get("hidden_dim", model_cfg.get("hidden_channels", 128)))
    num_filters = int(model_cfg.get("num_filters", hidden_dim))
    num_interactions = int(model_cfg.get("num_interactions", model_cfg.get("num_layers", 6)))
    num_rbf = int(model_cfg.get("num_rbf", model_cfg.get("num_radial", model_cfg.get("num_gaussians", 32))))
    cutoff = float(model_cfg.get("cutoff", model_cfg.get("radius_cutoff", 5.0)))
    max_num_neighbors = int(model_cfg.get("max_num_neighbors", model_cfg.get("max_neighbors", 32)))
    num_atom_types = int(model_cfg.get("num_atom_types", model_cfg.get("num_elements", model_cfg.get("max_z", 100))))
    num_features = int(model_cfg.get("num_features", model_cfg.get("in_dim", in_dim)))
    num_output_layers = int(model_cfg.get("num_output_layers", 2))
    readout = str(model_cfg.get("readout", "add"))
    backend = str(model_cfg.get("backend", "auto"))
    torchmd_model = str(model_cfg.get("torchmd_model", "equivariant-transformer"))
    activation = str(model_cfg.get("activation", "silu"))
    attn_activation = str(model_cfg.get("attn_activation", activation))
    trainable_rbf = bool(model_cfg.get("trainable_rbf", True))
    rbf_type = str(model_cfg.get("rbf_type", "expnorm"))
    neighbor_embedding = bool(model_cfg.get("neighbor_embedding", True))
    num_heads = int(model_cfg.get("num_heads", 8))
    distance_influence = str(model_cfg.get("distance_influence", "both"))
    vector_cutoff = bool(model_cfg.get("vector_cutoff", False))
    aggr = str(model_cfg.get("aggr", readout))
    return PaiNNRegressor(
        hidden_dim=hidden_dim,
        num_filters=num_filters,
        num_interactions=num_interactions,
        num_rbf=num_rbf,
        cutoff=cutoff,
        max_num_neighbors=max_num_neighbors,
        out_dim=out_dim,
        num_atom_types=num_atom_types,
        num_features=num_features,
        num_output_layers=num_output_layers,
        readout=readout,
        backend=backend,
        torchmd_model=torchmd_model,
        activation=activation,
        attn_activation=attn_activation,
        trainable_rbf=trainable_rbf,
        rbf_type=rbf_type,
        neighbor_embedding=neighbor_embedding,
        num_heads=num_heads,
        distance_influence=distance_influence,
        vector_cutoff=vector_cutoff,
        aggr=aggr,
    )


def _gnn_egnn_factory(model_cfg: Dict[str, Any], context: Dict[str, Any]):
    from src.gnn.models import EGNNRegressor

    in_dim = _require_context(context, "in_dim")
    edge_dim = _context_int(context, "edge_dim", 0)
    global_dim = _context_int(context, "global_dim", 0)
    out_dim = _context_int(context, "out_dim", 1)
    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    num_layers = int(model_cfg.get("num_layers", model_cfg.get("num_interactions", 4)))
    dropout = float(model_cfg.get("dropout", 0.1))
    use_edge_attr = bool(model_cfg.get("use_edge_attr", True))
    return EGNNRegressor(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        edge_dim=edge_dim,
        global_dim=global_dim,
        out_dim=out_dim,
        use_edge_attr=use_edge_attr,
    )


def _gnn_graphormer_factory(model_cfg: Dict[str, Any], context: Dict[str, Any]):
    from src.gnn.models import GraphormerRegressor

    in_dim = _require_context(context, "in_dim")
    global_dim = _context_int(context, "global_dim", 0)
    out_dim = _context_int(context, "out_dim", 1)
    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    num_layers = int(model_cfg.get("num_layers", 4))
    num_heads = int(model_cfg.get("num_heads", 8))
    dropout = float(model_cfg.get("dropout", 0.1))
    attn_dropout = float(model_cfg.get("attn_dropout", model_cfg.get("attention_dropout", dropout)))
    mlp_ratio = float(model_cfg.get("mlp_ratio", 4.0))
    max_distance = int(model_cfg.get("max_distance", 5))
    max_degree = int(model_cfg.get("max_degree", 10))
    prefer_pyg = bool(model_cfg.get("prefer_pyg", True))
    return GraphormerRegressor(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        attn_dropout=attn_dropout,
        mlp_ratio=mlp_ratio,
        max_distance=max_distance,
        max_degree=max_degree,
        global_dim=global_dim,
        out_dim=out_dim,
        prefer_pyg=prefer_pyg,
    )


def _smiles_transformer_factory(model_cfg: Dict[str, Any], context: Dict[str, Any]):
    from src.smiles.models import build_smiles_model

    return build_smiles_model(model_cfg=model_cfg, context=context)


_FP_CAPS = ModelCapabilities(
    supports_3d=False,
    supports_multitask=False,
    supports_force=False,
    requires_3d_pos=False,
    requires_distance_edges=False,
    accepts_global_descriptors=True,
    input_requires=(),
)

_GNN_CAPS = ModelCapabilities(
    supports_3d=False,
    supports_multitask=True,
    supports_force=False,
    requires_3d_pos=False,
    requires_distance_edges=False,
    accepts_global_descriptors=True,
    input_requires=(),
)

_MPNN_CAPS = ModelCapabilities(
    supports_3d=False,
    supports_multitask=True,
    supports_force=False,
    requires_3d_pos=False,
    requires_distance_edges=False,
    accepts_global_descriptors=True,
    input_requires=("edge_attr",),
)

_SCHNET_CAPS = ModelCapabilities(
    supports_3d=True,
    supports_multitask=False,
    supports_force=False,
    requires_3d_pos=True,
    requires_distance_edges=True,
    accepts_global_descriptors=False,
    input_requires=("pos", "edge_attr"),
)

_DIMENETPP_CAPS = ModelCapabilities(
    supports_3d=True,
    supports_multitask=False,
    supports_force=False,
    requires_3d_pos=True,
    requires_distance_edges=True,
    accepts_global_descriptors=False,
    input_requires=("pos", "angles"),
)

_PAINN_CAPS = ModelCapabilities(
    supports_3d=True,
    supports_multitask=False,
    supports_force=False,
    requires_3d_pos=True,
    requires_distance_edges=True,
    accepts_global_descriptors=False,
    input_requires=("pos", "edge_attr"),
)

_EGNN_CAPS = ModelCapabilities(
    supports_3d=True,
    supports_multitask=True,
    supports_force=False,
    requires_3d_pos=True,
    requires_distance_edges=False,
    accepts_global_descriptors=True,
    input_requires=("pos",),
)

_GRAPHORMER_CAPS = ModelCapabilities(
    supports_3d=False,
    supports_multitask=True,
    supports_force=False,
    requires_3d_pos=False,
    requires_distance_edges=False,
    accepts_global_descriptors=True,
    input_requires=(),
)

_SMILES_CAPS = ModelCapabilities(
    supports_3d=False,
    supports_multitask=True,
    supports_force=False,
    requires_3d_pos=False,
    requires_distance_edges=False,
    accepts_global_descriptors=False,
    input_requires=("tokens",),
)

for name in ["lightgbm", "lgbm", "rf", "random_forest", "catboost", "gpr"]:
    register_model(name=name, family="fp", factory=_fp_factory(name), capabilities=_FP_CAPS)

register_model(name="gcn", family="gnn", factory=_gnn_gcn_factory, capabilities=_GNN_CAPS)
register_model(name="gin", family="gnn", factory=_gnn_gin_factory, capabilities=_GNN_CAPS)
register_model(name="mpnn", family="gnn", factory=_gnn_mpnn_factory, capabilities=_MPNN_CAPS)
register_model(name="schnet", family="gnn", factory=_gnn_schnet_factory, capabilities=_SCHNET_CAPS)
register_model(name="dimenetpp", family="gnn", factory=_gnn_dimenetpp_factory, capabilities=_DIMENETPP_CAPS)
register_model(name="painn", family="gnn", factory=_gnn_painn_factory, capabilities=_PAINN_CAPS)
register_model(name="egnn", family="gnn", factory=_gnn_egnn_factory, capabilities=_EGNN_CAPS)
register_model(name="graphormer", family="gnn", factory=_gnn_graphormer_factory, capabilities=_GRAPHORMER_CAPS)
register_model(name="graph_transformer", family="gnn", factory=_gnn_graphormer_factory, capabilities=_GRAPHORMER_CAPS)

register_model(name="chemberta", family="smiles", factory=_smiles_transformer_factory, capabilities=_SMILES_CAPS)
register_model(name="smiles_transformer", family="smiles", factory=_smiles_transformer_factory, capabilities=_SMILES_CAPS)
