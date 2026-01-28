from __future__ import annotations

import inspect
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None

try:
    from torch_geometric.nn import GCNConv, GINConv, GINEConv, global_mean_pool, NNConv
except Exception:  # pragma: no cover
    GCNConv = None
    GINConv = None
    GINEConv = None
    global_mean_pool = None
    NNConv = None

try:
    from torch_geometric.nn.models import SchNet as PyGSchNet
except Exception:  # pragma: no cover
    try:
        from torch_geometric.nn import SchNet as PyGSchNet
    except Exception:  # pragma: no cover
        PyGSchNet = None

try:
    from torch_geometric.nn.models import DimeNetPlusPlus as PyGDimeNetPlusPlus
except Exception:  # pragma: no cover
    try:
        from torch_geometric.nn import DimeNetPlusPlus as PyGDimeNetPlusPlus
    except Exception:  # pragma: no cover
        PyGDimeNetPlusPlus = None

try:
    from torch_geometric.nn.models import PaiNN as PyGPaiNN
except Exception:  # pragma: no cover
    try:
        from torch_geometric.nn import PaiNN as PyGPaiNN
    except Exception:  # pragma: no cover
        PyGPaiNN = None

try:
    from torch_geometric.nn.models import Graphormer as PyGGraphormer
except Exception:  # pragma: no cover
    try:
        from torch_geometric.nn import Graphormer as PyGGraphormer
    except Exception:  # pragma: no cover
        PyGGraphormer = None

try:
    from torchmdnet.models.torchmd_et import TorchMD_ET as TorchMD_ET
    from torchmdnet.models.torchmd_gn import TorchMD_GN as TorchMD_GN
    from torchmdnet.models.output_modules import Scalar as TorchMDScalar
    from torchmdnet.models.output_modules import EquivariantScalar as TorchMDEquivariantScalar
except Exception:  # pragma: no cover
    TorchMD_ET = None
    TorchMD_GN = None
    TorchMDScalar = None
    TorchMDEquivariantScalar = None

try:
    from torch_geometric.typing import WITH_TORCH_CLUSTER, WITH_TORCH_SCATTER
except Exception:  # pragma: no cover
    WITH_TORCH_CLUSTER = None
    WITH_TORCH_SCATTER = None


def _require_pyg():
    if torch is None or nn is None or F is None:
        raise ImportError("PyTorch is required for GNN models. Please install torch.")
    if global_mean_pool is None:
        raise ImportError(
            "PyTorch Geometric is required for GNN models. "
            "Please install torch_geometric (matching your torch/CUDA)."
        )

if nn is None:  # pragma: no cover

    class GCNRegressor:
        def __init__(self, *args, **kwargs):
            _require_pyg()

    class GINRegressor:
        def __init__(self, *args, **kwargs):
            _require_pyg()

    class MPNNRegressor:
        def __init__(self, *args, **kwargs):
            _require_pyg()

    class SchNetRegressor:
        def __init__(self, *args, **kwargs):
            _require_pyg()

    class DimeNetPPRegressor:
        def __init__(self, *args, **kwargs):
            _require_pyg()

    class PaiNNRegressor:
        def __init__(self, *args, **kwargs):
            _require_pyg()

    class EGNNRegressor:
        def __init__(self, *args, **kwargs):
            _require_pyg()

    class GraphormerRegressor:
        def __init__(self, *args, **kwargs):
            _require_pyg()

else:

    class GCNRegressor(nn.Module):
        def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float = 0.0,
            global_dim: int = 0,
            out_dim: int = 1,
        ):
            super().__init__()
            _require_pyg()
            if GCNConv is None:
                raise ImportError("GCNConv is unavailable. Please install torch_geometric.")
            self.dropout = float(dropout)
            self.convs = nn.ModuleList()
            self.convs.append(GCNConv(in_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.head = nn.Sequential(
                nn.Linear(hidden_dim + global_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, int(out_dim)),
            )

        def encode(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            for conv in self.convs:
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            g = global_mean_pool(x, batch)
            if hasattr(data, "u"):
                # data.u shape (batch, global_dim)
                g = torch.cat([g, data.u], dim=-1)
            return g

        def forward(self, data):
            g = self.encode(data)
            return self.head(g)

    class MPNNRegressor(nn.Module):
        """
        A simple message passing model using NNConv with edge-conditioned filters.
        """

        def __init__(
            self,
            in_dim: int,
            edge_dim: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float = 0.0,
            global_dim: int = 0,
            edge_mlp_hidden_dim: int = 128,
            out_dim: int = 1,
        ):
            super().__init__()
            _require_pyg()
            if NNConv is None:
                raise ImportError("NNConv is unavailable. Please install torch_geometric.")
            self.dropout = float(dropout)

            if edge_mlp_hidden_dim <= 0:
                raise ValueError("edge_mlp_hidden_dim must be > 0")

            def make_edge_nn(out_dim: int) -> nn.Module:
                # Map edge_attr -> (in_channels * out_channels) weight matrix for NNConv.
                return nn.Sequential(
                    nn.Linear(edge_dim, edge_mlp_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(edge_mlp_hidden_dim, out_dim),
                )

            self.conv1 = NNConv(in_dim, hidden_dim, make_edge_nn(hidden_dim * in_dim), aggr="mean")

            self.convs = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.convs.append(NNConv(hidden_dim, hidden_dim, make_edge_nn(hidden_dim * hidden_dim), aggr="mean"))

            self.head = nn.Sequential(
                nn.Linear(hidden_dim + global_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, int(out_dim)),
            )

        def encode(self, data):
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
            x = self.conv1(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            for conv in self.convs:
                x = conv(x, edge_index, edge_attr)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

            g = global_mean_pool(x, batch)
            if hasattr(data, "u"):
                g = torch.cat([g, data.u], dim=-1)
            return g

        def forward(self, data):
            g = self.encode(data)
            return self.head(g)

    class GINRegressor(nn.Module):
        """
        Graph Isomorphism Network (GIN/GINE) regressor.
        """

        def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float = 0.0,
            global_dim: int = 0,
            edge_dim: int = 0,
            out_dim: int = 1,
        ):
            super().__init__()
            _require_pyg()
            if GINConv is None:
                raise ImportError("GINConv is unavailable. Please install torch_geometric.")
            self.dropout = float(dropout)
            self.use_edge_attr = edge_dim > 0 and GINEConv is not None

            def make_mlp(input_dim: int) -> nn.Module:
                return nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )

            self.convs = nn.ModuleList()
            if self.use_edge_attr:
                self.convs.append(GINEConv(make_mlp(in_dim), edge_dim=edge_dim))
                for _ in range(num_layers - 1):
                    self.convs.append(GINEConv(make_mlp(hidden_dim), edge_dim=edge_dim))
            else:
                self.convs.append(GINConv(make_mlp(in_dim)))
                for _ in range(num_layers - 1):
                    self.convs.append(GINConv(make_mlp(hidden_dim)))

            self.head = nn.Sequential(
                nn.Linear(hidden_dim + global_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, int(out_dim)),
            )

        def encode(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            edge_attr = getattr(data, "edge_attr", None)
            for conv in self.convs:
                if self.use_edge_attr:
                    x = conv(x, edge_index, edge_attr)
                else:
                    x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            g = global_mean_pool(x, batch)
            if hasattr(data, "u"):
                g = torch.cat([g, data.u], dim=-1)
            return g

        def forward(self, data):
            g = self.encode(data)
            return self.head(g)

    class _GraphMultiheadAttention(nn.Module):
        def __init__(self, hidden_dim: int, num_heads: int, dropout: float, attn_dropout: float) -> None:
            super().__init__()
            if hidden_dim % num_heads != 0:
                raise ValueError("hidden_dim must be divisible by num_heads.")
            self.hidden_dim = int(hidden_dim)
            self.num_heads = int(num_heads)
            self.head_dim = int(hidden_dim // num_heads)
            self.scale = self.head_dim**-0.5
            self.qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3)
            self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.dropout = nn.Dropout(float(dropout))
            self.attn_dropout = nn.Dropout(float(attn_dropout))

        def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor]) -> torch.Tensor:
            num_nodes = x.size(0)
            qkv = self.qkv(x)
            qkv = qkv.view(num_nodes, 3, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if attn_bias is not None:
                attn_scores = attn_scores + attn_bias
            attn = F.softmax(attn_scores, dim=-1)
            attn = self.attn_dropout(attn)
            out = torch.matmul(attn, v)
            out = out.transpose(0, 1).contiguous().view(num_nodes, self.hidden_dim)
            out = self.out_proj(out)
            return self.dropout(out)

    class _GraphTransformerLayer(nn.Module):
        def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            dropout: float,
            attn_dropout: float,
            mlp_ratio: float,
        ) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.attn = _GraphMultiheadAttention(hidden_dim, num_heads, dropout, attn_dropout)
            self.norm2 = nn.LayerNorm(hidden_dim)
            mlp_hidden = int(hidden_dim * float(mlp_ratio))
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, mlp_hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(mlp_hidden, hidden_dim),
                nn.Dropout(float(dropout)),
            )

        def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor]) -> torch.Tensor:
            x = x + self.attn(self.norm1(x), attn_bias=attn_bias)
            x = x + self.ffn(self.norm2(x))
            return x

    class GraphormerRegressor(nn.Module):
        """
        Graphormer-style regressor with shortest-path attention bias.
        """

        def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            num_layers: int,
            num_heads: int,
            dropout: float = 0.1,
            attn_dropout: float = 0.1,
            mlp_ratio: float = 4.0,
            max_distance: int = 5,
            max_degree: int = 10,
            global_dim: int = 0,
            out_dim: int = 1,
            prefer_pyg: bool = True,
        ) -> None:
            super().__init__()
            _require_pyg()
            self.hidden_dim = int(hidden_dim)
            self.max_distance = int(max_distance)
            self.max_degree = int(max_degree)
            self.global_dim = int(global_dim)
            self.out_dim = int(out_dim)
            if self.max_distance < 1:
                raise ValueError("max_distance must be >= 1")
            if self.max_degree < 1:
                raise ValueError("max_degree must be >= 1")

            self._pyg_backend = None
            self._pyg_forward_params: set[str] = set()
            self._use_pyg = False

            self.input_proj = nn.Linear(int(in_dim), self.hidden_dim)
            self.input_dropout = nn.Dropout(float(dropout))
            self.degree_emb = nn.Embedding(self.max_degree + 2, self.hidden_dim)
            self.spatial_pos_emb = nn.Embedding(self.max_distance + 2, int(num_heads))
            self.layers = nn.ModuleList(
                [
                    _GraphTransformerLayer(
                        hidden_dim=self.hidden_dim,
                        num_heads=int(num_heads),
                        dropout=float(dropout),
                        attn_dropout=float(attn_dropout),
                        mlp_ratio=float(mlp_ratio),
                    )
                    for _ in range(int(num_layers))
                ]
            )
            self.norm = nn.LayerNorm(self.hidden_dim)
            self.head = nn.Sequential(
                nn.Linear(self.hidden_dim + self.global_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.out_dim),
            )

            if prefer_pyg and PyGGraphormer is not None:
                backend = self._init_pyg_backend(
                    in_dim=in_dim,
                    hidden_dim=hidden_dim,
                    out_dim=out_dim,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                if backend is not None:
                    self._pyg_backend = backend
                    self._use_pyg = True

        @staticmethod
        def _build_neighbors(edge_index: torch.Tensor, num_nodes: int) -> list[set[int]]:
            neighbors = [set() for _ in range(num_nodes)]
            if edge_index is None or edge_index.numel() == 0 or num_nodes == 0:
                return neighbors
            edge_index_cpu = edge_index.detach().cpu()
            for idx in range(edge_index_cpu.shape[1]):
                src = int(edge_index_cpu[0, idx])
                dst = int(edge_index_cpu[1, idx])
                if src == dst:
                    continue
                if 0 <= src < num_nodes and 0 <= dst < num_nodes:
                    neighbors[src].add(dst)
                    neighbors[dst].add(src)
            return neighbors

        def _shortest_path_distances(
            self,
            neighbors: list[set[int]],
            num_nodes: int,
            device: torch.device,
        ) -> torch.Tensor:
            max_dist = self.max_distance
            dist = torch.full((num_nodes, num_nodes), max_dist + 1, dtype=torch.long)
            for i in range(num_nodes):
                dist[i, i] = 0
                visited = {i}
                frontier = [i]
                for d in range(1, max_dist + 1):
                    next_frontier = []
                    for u in frontier:
                        for v in neighbors[u]:
                            if v in visited:
                                continue
                            visited.add(v)
                            dist[i, v] = d
                            next_frontier.append(v)
                    if not next_frontier:
                        break
                    frontier = next_frontier
            return dist.to(device=device)

        def _build_attn_bias(
            self,
            edge_index: torch.Tensor,
            num_nodes: int,
            device: torch.device,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            neighbors = self._build_neighbors(edge_index, num_nodes)
            degrees = torch.tensor([len(n) for n in neighbors], dtype=torch.long, device=device)
            degree_bucket = torch.clamp(degrees, max=self.max_degree + 1)
            spatial_pos = self._shortest_path_distances(neighbors, num_nodes, device)
            spatial_pos = torch.where(spatial_pos > self.max_distance, self.max_distance + 1, spatial_pos)
            attn_bias = self.spatial_pos_emb(spatial_pos).permute(2, 0, 1)
            return attn_bias, degree_bucket, spatial_pos

        @staticmethod
        def _init_backend(model_cls, kwargs):
            try:
                sig = inspect.signature(model_cls.__init__)
            except Exception:
                return model_cls(**kwargs)
            params = sig.parameters
            if any(p.kind == p.VAR_KEYWORD for p in params.values()):
                return model_cls(**kwargs)
            filtered = {k: v for k, v in kwargs.items() if k in params}
            required = [
                name
                for name, param in params.items()
                if name != "self"
                and param.default is param.empty
                and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
            ]
            missing = [name for name in required if name not in filtered]
            if missing:
                raise ValueError(f"Graphormer backend requires parameters: {', '.join(missing)}")
            return model_cls(**filtered)

        def _init_pyg_backend(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int,
            num_layers: int,
            num_heads: int,
            dropout: float,
            attn_dropout: float,
        ):
            init_kwargs = {
                "in_channels": int(in_dim),
                "hidden_channels": int(hidden_dim),
                "out_channels": int(out_dim),
                "num_layers": int(num_layers),
                "num_heads": int(num_heads),
                "dropout": float(dropout),
                "attn_dropout": float(attn_dropout),
            }
            try:
                backend = self._init_backend(PyGGraphormer, init_kwargs)
            except Exception:
                return None
            try:
                sig = inspect.signature(backend.forward)
                self._pyg_forward_params = set(sig.parameters.keys())
            except Exception:
                self._pyg_forward_params = set()
            return backend

        def _forward_pyg(self, data):
            if self._pyg_backend is None:
                raise RuntimeError("PyG Graphormer backend is not initialized.")
            try:
                return self._pyg_backend(data)
            except Exception:
                params = self._pyg_forward_params or set()
                kwargs = {}
                if "x" in params:
                    kwargs["x"] = data.x
                if "edge_index" in params:
                    kwargs["edge_index"] = data.edge_index
                if "edge_attr" in params and hasattr(data, "edge_attr"):
                    kwargs["edge_attr"] = data.edge_attr
                if "batch" in params and hasattr(data, "batch"):
                    kwargs["batch"] = data.batch
                if "ptr" in params and hasattr(data, "ptr"):
                    kwargs["ptr"] = data.ptr
                needs_struct = any(
                    key in params for key in {"attn_bias", "spatial_pos", "in_degree", "out_degree"}
                )
                if needs_struct:
                    attn_bias, degree_bucket, spatial_pos = self._build_attn_bias(
                        data.edge_index, int(data.x.size(0)), data.x.device
                    )
                    if "attn_bias" in params:
                        kwargs["attn_bias"] = attn_bias
                    if "spatial_pos" in params:
                        kwargs["spatial_pos"] = spatial_pos
                    if "in_degree" in params:
                        kwargs["in_degree"] = degree_bucket
                    if "out_degree" in params:
                        kwargs["out_degree"] = degree_bucket
                if not kwargs and params:
                    raise RuntimeError("Unable to map Graphormer inputs for backend forward.")
                return self._pyg_backend(**kwargs)

        def _encode_graph(self, data) -> torch.Tensor:
            x = data.x
            if x.dim() == 1:
                x = x.view(-1, 1)
            num_nodes = int(x.size(0))
            attn_bias, degree_bucket, _ = self._build_attn_bias(data.edge_index, num_nodes, x.device)
            h = self.input_proj(x)
            h = self.input_dropout(h)
            h = h + self.degree_emb(degree_bucket)
            for layer in self.layers:
                h = layer(h, attn_bias=attn_bias)
            h = self.norm(h)
            g = h.mean(dim=0, keepdim=True)
            if hasattr(data, "u"):
                u = data.u
                if u.dim() == 1:
                    u = u.view(1, -1)
                g = torch.cat([g, u], dim=-1)
            return g

        def _forward_graph(self, data) -> torch.Tensor:
            g = self._encode_graph(data)
            return self.head(g)

        def encode(self, data) -> torch.Tensor:
            if hasattr(data, "batch") and hasattr(data, "num_graphs"):
                data_list = data.to_data_list()
                out_list = [self._encode_graph(item) for item in data_list]
                return torch.cat(out_list, dim=0)
            return self._encode_graph(data)

        def forward(self, data):
            if self._pyg_backend is not None and self._use_pyg:
                try:
                    return self._forward_pyg(data)
                except Exception:
                    self._use_pyg = False
            if hasattr(data, "batch") and hasattr(data, "num_graphs"):
                data_list = data.to_data_list()
                out_list = [self._forward_graph(item) for item in data_list]
                return torch.cat(out_list, dim=0)
            return self._forward_graph(data)

    class _EGNNLayer(nn.Module):
        def __init__(self, hidden_dim: int, edge_dim: int, dropout: float, use_edge_attr: bool) -> None:
            super().__init__()
            self.hidden_dim = int(hidden_dim)
            self.edge_dim = int(edge_dim)
            self.use_edge_attr = bool(use_edge_attr and self.edge_dim > 0)
            msg_in = self.hidden_dim * 2 + 1 + (self.edge_dim if self.use_edge_attr else 0)
            self.edge_mlp = nn.Sequential(
                nn.Linear(msg_in, self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
            )
            self.pos_mlp = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 1),
            )
            self.node_mlp = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )

        @staticmethod
        def _scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
            out_shape = (int(dim_size),) + src.shape[1:]
            out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
            if src.numel() == 0:
                return out
            out.index_add_(0, index, src)
            return out

        def forward(
            self,
            h: torch.Tensor,
            pos: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: Optional[torch.Tensor] = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if edge_index is None or edge_index.numel() == 0:
                agg = torch.zeros_like(h)
                h = h + self.node_mlp(torch.cat([h, agg], dim=-1))
                return h, pos
            row, col = edge_index[0], edge_index[1]
            rel = pos[row] - pos[col]
            dist2 = (rel * rel).sum(dim=-1, keepdim=True)
            if self.use_edge_attr and edge_attr is not None:
                msg_in = torch.cat([h[row], h[col], dist2, edge_attr], dim=-1)
            else:
                msg_in = torch.cat([h[row], h[col], dist2], dim=-1)
            msg = self.edge_mlp(msg_in)
            delta = rel * self.pos_mlp(msg)
            pos = pos + self._scatter_add(delta, row, dim_size=pos.size(0))
            agg = self._scatter_add(msg, row, dim_size=h.size(0))
            h = h + self.node_mlp(torch.cat([h, agg], dim=-1))
            return h, pos

    class EGNNRegressor(nn.Module):
        """
        Lightweight EGNN-style regressor for graph-level prediction.
        """

        def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float = 0.0,
            edge_dim: int = 0,
            global_dim: int = 0,
            out_dim: int = 1,
            use_edge_attr: bool = True,
        ):
            super().__init__()
            _require_pyg()
            if in_dim <= 0:
                raise ValueError("EGNN requires in_dim > 0 (node features).")
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be > 0")
            if num_layers <= 0:
                raise ValueError("num_layers must be > 0")
            if out_dim <= 0:
                raise ValueError("out_dim must be > 0")
            self.edge_dim = int(edge_dim)
            self.use_edge_attr = bool(use_edge_attr)
            self.global_dim = int(global_dim)
            self.out_dim = int(out_dim)
            self.input_proj = nn.Linear(int(in_dim), int(hidden_dim))
            self.layers = nn.ModuleList(
                [
                    _EGNNLayer(
                        hidden_dim=int(hidden_dim),
                        edge_dim=self.edge_dim,
                        dropout=float(dropout),
                        use_edge_attr=self.use_edge_attr,
                    )
                    for _ in range(int(num_layers))
                ]
            )
            self.head = nn.Sequential(
                nn.Linear(int(hidden_dim) + self.global_dim, int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), self.out_dim),
            )

        def forward(self, data):
            if not hasattr(data, "pos"):
                raise ValueError("EGNN requires data.pos (3D positions).")
            if not hasattr(data, "edge_index"):
                raise ValueError("EGNN requires data.edge_index.")
            x = getattr(data, "x", None)
            if x is None:
                if not hasattr(data, "z"):
                    raise ValueError("EGNN requires data.x or data.z.")
                x = data.z
            if x.dim() == 1:
                x = x.view(-1, 1)
            h = self.input_proj(x.float())
            pos = data.pos
            edge_index = data.edge_index
            edge_attr = getattr(data, "edge_attr", None)
            if self.use_edge_attr and self.edge_dim > 0 and edge_attr is None:
                raise ValueError("EGNN requires data.edge_attr (edge features).")
            for layer in self.layers:
                h, pos = layer(h, pos, edge_index, edge_attr if self.use_edge_attr else None)
            batch = getattr(data, "batch", None)
            if batch is None:
                batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            g = global_mean_pool(h, batch)
            if hasattr(data, "u"):
                u = data.u
                if u.dim() == 1:
                    u = u.view(1, -1)
                g = torch.cat([g, u], dim=-1)
            out = self.head(g)
            if out.dim() == 2 and out.shape[1] == 1:
                return out.view(-1)
            return out

    class SchNetRegressor(nn.Module):
        def __init__(
            self,
            hidden_dim: int,
            num_filters: int,
            num_interactions: int,
            num_gaussians: int,
            cutoff: float,
            readout: str = "add",
        ):
            super().__init__()
            _require_pyg()
            if PyGSchNet is None:
                raise ImportError("SchNet is unavailable. Please install torch_geometric with SchNet support.")
            if WITH_TORCH_CLUSTER is False:
                raise ImportError(
                    "SchNet requires torch_cluster for radius graph. Please install torch-cluster."
                )
            if WITH_TORCH_CLUSTER is None:
                try:
                    import torch_cluster  # noqa: F401
                except Exception as exc:
                    raise ImportError(
                        "SchNet requires torch_cluster for radius graph. Please install torch-cluster."
                    ) from exc
            if num_gaussians <= 0:
                raise ValueError("num_gaussians must be > 0")
            self.model = PyGSchNet(
                hidden_channels=int(hidden_dim),
                num_filters=int(num_filters),
                num_interactions=int(num_interactions),
                num_gaussians=int(num_gaussians),
                cutoff=float(cutoff),
                readout=str(readout),
            )

        def forward(self, data):
            if not hasattr(data, "z"):
                raise ValueError("SchNet requires data.z (atomic numbers).")
            if not hasattr(data, "pos"):
                raise ValueError("SchNet requires data.pos (3D positions).")
            z = data.z
            if z.dtype != torch.long:
                z = z.long()
            pos = data.pos
            batch = getattr(data, "batch", None)
            if batch is None:
                batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
            out = self.model(z, pos, batch)
            return out.view(-1)

    class DimeNetPPRegressor(nn.Module):
        def __init__(
            self,
            hidden_dim: int,
            out_dim: int,
            num_blocks: int,
            num_bilinear: int,
            num_spherical: int,
            num_radial: int,
            cutoff: float,
            max_num_neighbors: int,
            envelope_exponent: int,
            num_before_skip: int,
            num_after_skip: int,
            num_output_layers: int,
            int_emb_size: Optional[int] = None,
            basis_emb_size: Optional[int] = None,
            out_emb_channels: Optional[int] = None,
        ):
            super().__init__()
            _require_pyg()
            if PyGDimeNetPlusPlus is None:
                raise ImportError(
                    "DimeNet++ is unavailable. Install torch_geometric with DimeNet++ support "
                    "and optional deps (e.g. `pip install -e .[3d]`)."
                )
            if WITH_TORCH_CLUSTER is False:
                raise ImportError(
                    "DimeNet++ requires torch_cluster for radius graph. Please install torch-cluster."
                )
            if WITH_TORCH_CLUSTER is None:
                try:
                    import torch_cluster  # noqa: F401
                except Exception as exc:
                    raise ImportError(
                        "DimeNet++ requires torch_cluster for radius graph. Please install torch-cluster."
                    ) from exc
            if WITH_TORCH_SCATTER is False:
                raise ImportError(
                    "DimeNet++ requires torch_scatter for message passing. Please install torch-scatter."
                )
            if WITH_TORCH_SCATTER is None:
                try:
                    import torch_scatter  # noqa: F401
                except Exception as exc:
                    raise ImportError(
                        "DimeNet++ requires torch_scatter for message passing. Please install torch-scatter."
                    ) from exc
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be > 0")
            if out_dim <= 0:
                raise ValueError("out_dim must be > 0")
            import inspect

            sig = inspect.signature(PyGDimeNetPlusPlus)
            if "num_bilinear" in sig.parameters:
                self.model = PyGDimeNetPlusPlus(
                    hidden_channels=int(hidden_dim),
                    out_channels=int(out_dim),
                    num_blocks=int(num_blocks),
                    num_bilinear=int(num_bilinear),
                    num_spherical=int(num_spherical),
                    num_radial=int(num_radial),
                    cutoff=float(cutoff),
                    max_num_neighbors=int(max_num_neighbors),
                    envelope_exponent=int(envelope_exponent),
                    num_before_skip=int(num_before_skip),
                    num_after_skip=int(num_after_skip),
                    num_output_layers=int(num_output_layers),
                )
            else:
                int_emb = int(int_emb_size or hidden_dim)
                basis_emb = int(basis_emb_size or hidden_dim)
                out_emb = int(out_emb_channels or hidden_dim)
                self.model = PyGDimeNetPlusPlus(
                    hidden_channels=int(hidden_dim),
                    out_channels=int(out_dim),
                    num_blocks=int(num_blocks),
                    int_emb_size=int_emb,
                    basis_emb_size=basis_emb,
                    out_emb_channels=out_emb,
                    num_spherical=int(num_spherical),
                    num_radial=int(num_radial),
                    cutoff=float(cutoff),
                    max_num_neighbors=int(max_num_neighbors),
                    envelope_exponent=int(envelope_exponent),
                    num_before_skip=int(num_before_skip),
                    num_after_skip=int(num_after_skip),
                    num_output_layers=int(num_output_layers),
                )

        def forward(self, data):
            if not hasattr(data, "z"):
                raise ValueError("DimeNet++ requires data.z (atomic numbers).")
            if not hasattr(data, "pos"):
                raise ValueError("DimeNet++ requires data.pos (3D positions).")
            z = data.z
            if z.dtype != torch.long:
                z = z.long()
            pos = data.pos
            batch = getattr(data, "batch", None)
            if batch is None:
                batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
            out = self.model(z, pos, batch)
            if out.dim() == 2 and out.shape[1] == 1:
                return out.view(-1)
            return out

    class PaiNNRegressor(nn.Module):
        def __init__(
            self,
            hidden_dim: int,
            num_filters: int,
            num_interactions: int,
            num_rbf: int,
            cutoff: float,
            max_num_neighbors: int,
            out_dim: int = 1,
            num_atom_types: int = 100,
            num_features: int = 0,
            num_output_layers: int = 2,
            readout: str = "add",
            backend: str = "auto",
            torchmd_model: str = "equivariant-transformer",
            activation: str = "silu",
            attn_activation: str = "silu",
            trainable_rbf: bool = True,
            rbf_type: str = "expnorm",
            neighbor_embedding: bool = True,
            num_heads: int = 8,
            distance_influence: str = "both",
            vector_cutoff: bool = False,
            aggr: str = "add",
        ):
            super().__init__()
            self.backend = None
            self.output_model = None
            self.model = None
            self._forward_params = None
            backend = str(backend or "auto").lower()
            if backend in {"", "auto"}:
                use_pyg = PyGPaiNN is not None
                use_torchmdnet = not use_pyg
            elif backend in {"pyg", "torch_geometric", "torch-geometric"}:
                use_pyg = True
                use_torchmdnet = False
            elif backend in {"torchmdnet", "torchmd"}:
                use_pyg = False
                use_torchmdnet = True
            else:
                raise ValueError(f"Unknown PaiNN backend: {backend}")

            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be > 0")
            if num_filters <= 0:
                raise ValueError("num_filters must be > 0")
            if num_interactions <= 0:
                raise ValueError("num_interactions must be > 0")
            if num_rbf <= 0:
                raise ValueError("num_rbf must be > 0")
            if out_dim <= 0:
                raise ValueError("out_dim must be > 0")
            if cutoff <= 0:
                raise ValueError("cutoff must be > 0")
            if max_num_neighbors <= 0:
                raise ValueError("max_num_neighbors must be > 0")
            if num_atom_types <= 0:
                raise ValueError("num_atom_types must be > 0")
            if use_torchmdnet:
                if torch is None or nn is None:
                    raise ImportError("PyTorch is required for the TorchMD-Net PaiNN backend.")
                if TorchMD_ET is None and TorchMD_GN is None:
                    raise ImportError(
                        "TorchMD-Net backend is unavailable. Install torchmd-net "
                        "(e.g. from https://github.com/torchmd/torchmd-net.git)."
                    )
                if out_dim != 1:
                    raise ValueError("TorchMD-Net PaiNN backend supports a single regression target.")
                reduce_map = {"add": "sum", "sum": "sum", "mean": "mean", "max": "max"}
                reduce_op = reduce_map.get(str(readout).lower())
                if reduce_op is None:
                    raise ValueError(f"Unsupported readout for TorchMD-Net backend: {readout}")
                torchmd_model = str(torchmd_model or "equivariant-transformer").lower()
                activation = str(activation or "silu")
                attn_activation = str(attn_activation or activation)
                rbf_type = str(rbf_type or "expnorm")
                neighbor_embedding = bool(neighbor_embedding)
                num_heads = int(num_heads)
                distance_influence = str(distance_influence or "both")
                vector_cutoff = bool(vector_cutoff)
                aggr = str(aggr or readout).lower()
                if aggr == "sum":
                    aggr = "add"
                if aggr not in {"add", "mean", "max"}:
                    raise ValueError(f"Unsupported TorchMD-Net aggr: {aggr}")
                if torchmd_model in {"equivariant-transformer", "equivariant_transformer", "et"}:
                    if TorchMD_ET is None or TorchMDEquivariantScalar is None:
                        raise ImportError("TorchMD-Net equivariant transformer backend is unavailable.")
                    self.model = TorchMD_ET(
                        hidden_channels=int(hidden_dim),
                        num_layers=int(num_interactions),
                        num_rbf=int(num_rbf),
                        rbf_type=rbf_type,
                        trainable_rbf=bool(trainable_rbf),
                        activation=activation,
                        attn_activation=attn_activation,
                        neighbor_embedding=neighbor_embedding,
                        num_heads=num_heads,
                        distance_influence=distance_influence,
                        cutoff_lower=0.0,
                        cutoff_upper=float(cutoff),
                        max_z=int(num_atom_types),
                        max_num_neighbors=int(max_num_neighbors),
                        vector_cutoff=vector_cutoff,
                    )
                    self.output_model = TorchMDEquivariantScalar(
                        int(hidden_dim),
                        activation=activation,
                        reduce_op=reduce_op,
                        dtype=torch.float32,
                        num_layers=int(num_output_layers),
                    )
                elif torchmd_model in {"graph-network", "graph_network", "gn"}:
                    if TorchMD_GN is None or TorchMDScalar is None:
                        raise ImportError("TorchMD-Net graph-network backend is unavailable.")
                    self.model = TorchMD_GN(
                        hidden_channels=int(hidden_dim),
                        num_filters=int(num_filters),
                        num_layers=int(num_interactions),
                        num_rbf=int(num_rbf),
                        rbf_type=rbf_type,
                        trainable_rbf=bool(trainable_rbf),
                        activation=activation,
                        neighbor_embedding=neighbor_embedding,
                        cutoff_lower=0.0,
                        cutoff_upper=float(cutoff),
                        max_z=int(num_atom_types),
                        max_num_neighbors=int(max_num_neighbors),
                        aggr=aggr,
                    )
                    self.output_model = TorchMDScalar(
                        int(hidden_dim),
                        activation=activation,
                        reduce_op=reduce_op,
                        dtype=torch.float32,
                        num_layers=int(num_output_layers),
                    )
                else:
                    raise ValueError(f"Unknown TorchMD-Net model: {torchmd_model}")
                self.backend = "torchmdnet"
            else:
                _require_pyg()
                if PyGPaiNN is None:
                    raise ImportError(
                        "PaiNN is unavailable in torch_geometric. "
                        "Install torch_geometric with PaiNN support or use backend='torchmdnet'."
                    )
                if WITH_TORCH_CLUSTER is False:
                    raise ImportError(
                        "PaiNN requires torch_cluster for radius graph. Please install torch-cluster."
                    )
                if WITH_TORCH_CLUSTER is None:
                    try:
                        import torch_cluster  # noqa: F401
                    except Exception as exc:
                        raise ImportError(
                            "PaiNN requires torch_cluster for radius graph. Please install torch-cluster."
                        ) from exc
                if WITH_TORCH_SCATTER is False:
                    raise ImportError(
                        "PaiNN requires torch_scatter for message passing. Please install torch-scatter."
                    )
                if WITH_TORCH_SCATTER is None:
                    try:
                        import torch_scatter  # noqa: F401
                    except Exception as exc:
                        raise ImportError(
                            "PaiNN requires torch_scatter for message passing. Please install torch-scatter."
                        ) from exc
                init_kwargs = {
                    "hidden_dim": int(hidden_dim),
                    "hidden_channels": int(hidden_dim),
                    "num_filters": int(num_filters),
                    "num_channels": int(hidden_dim),
                    "num_interactions": int(num_interactions),
                    "num_rbf": int(num_rbf),
                    "num_radial": int(num_rbf),
                    "num_gaussians": int(num_rbf),
                    "cutoff": float(cutoff),
                    "max_num_neighbors": int(max_num_neighbors),
                    "out_dim": int(out_dim),
                    "out_channels": int(out_dim),
                    "num_output_layers": int(num_output_layers),
                    "readout": str(readout),
                }
                if num_atom_types > 0:
                    init_kwargs.update(
                        {
                            "num_atom_types": int(num_atom_types),
                            "num_elements": int(num_atom_types),
                            "max_z": int(num_atom_types),
                        }
                    )
                if num_features > 0:
                    init_kwargs.update(
                        {
                            "num_features": int(num_features),
                            "in_dim": int(num_features),
                            "in_channels": int(num_features),
                        }
                    )
                self.model = self._init_backend(PyGPaiNN, init_kwargs)
                self._forward_params = None
                self.backend = "pyg"

        @staticmethod
        def _init_backend(model_cls, kwargs):
            try:
                sig = inspect.signature(model_cls.__init__)
            except Exception:
                return model_cls(**kwargs)
            params = sig.parameters
            if any(p.kind == p.VAR_KEYWORD for p in params.values()):
                return model_cls(**kwargs)
            filtered = {k: v for k, v in kwargs.items() if k in params}
            required = [
                name
                for name, param in params.items()
                if name != "self"
                and param.default is param.empty
                and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
            ]
            missing = [name for name in required if name not in filtered]
            if missing:
                missing_str = ", ".join(missing)
                raise ValueError(
                    "PaiNN backend requires parameters: "
                    f"{missing_str}. Provide them via model config."
                )
            return model_cls(**filtered)

        def _resolve_forward_params(self):
            if self._forward_params is None:
                try:
                    sig = inspect.signature(self.model.forward)
                    self._forward_params = set(sig.parameters.keys())
                except Exception:
                    self._forward_params = set()
            return self._forward_params

        def forward(self, data):
            if not hasattr(data, "z"):
                raise ValueError("PaiNN requires data.z (atomic numbers).")
            if not hasattr(data, "pos"):
                raise ValueError("PaiNN requires data.pos (3D positions).")
            z = data.z
            if z.dtype != torch.long:
                z = z.long()
            pos = data.pos
            batch = getattr(data, "batch", None)
            if batch is None:
                batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
            if self.backend == "torchmdnet":
                x, v, z, pos, batch = self.model(z, pos, batch)
                out = self.output_model.pre_reduce(x, v, z, pos, batch)
                out = self.output_model.reduce(out, batch)
                out = self.output_model.post_reduce(out)
            else:
                edge_index = getattr(data, "edge_index", None)
                edge_attr = getattr(data, "edge_attr", None)
                params = self._resolve_forward_params()
                if params:
                    kwargs = {}
                    if "z" in params:
                        kwargs["z"] = z
                    elif "x" in params:
                        kwargs["x"] = data.x
                    if "pos" in params:
                        kwargs["pos"] = pos
                    if "edge_index" in params:
                        if edge_index is None:
                            raise ValueError("PaiNN requires data.edge_index.")
                        kwargs["edge_index"] = edge_index
                    if "edge_attr" in params:
                        if edge_attr is None:
                            raise ValueError("PaiNN requires data.edge_attr.")
                        kwargs["edge_attr"] = edge_attr
                    if "batch" in params:
                        kwargs["batch"] = batch
                    out = self.model(**kwargs)
                else:
                    out = self.model(z, pos, batch)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if out.dim() == 2 and out.shape[1] == 1:
                return out.view(-1)
            return out
