# QM9 Quick Benchmark Report (Baseline vs Improved)

Dataset: QM9 quick subset (1992 rows; train=1593, val=199, test=200).
Target: gap (eV).

## Baseline (previous config)
|entry|model|features|val_mae|val_rmse|val_r2|test_mae|test_rmse|test_r2|
|---|---|---|---|---|---|---|---|---|
|fp_morgan_desc_lgbm|fp_lightgbm_quick|fp_morgan_desc|59.290|109.806|0.991|120.174|354.286|0.931|
|gcn_graph_quick|gnn_gcn_quick|gnn_graph_quick|840.156|1044.835|0.204|853.916|1113.307|0.314|
|mpnn_graph_quick|gnn_mpnn_quick|gnn_graph_quick|658.380|896.635|0.414|670.170|944.675|0.506|
|smiles_chemberta|chemberta|smiles_tokenizer|8555.749|8635.557|-53.343|8577.017|8681.670|-40.725|

## Improved (z-score target + 3D radius GNN + SMILES tweaks)
|entry|model|features|val_mae|val_rmse|val_r2|test_mae|test_rmse|test_r2|
|---|---|---|---|---|---|---|---|---|
|fp_morgan_desc_lgbm|fp_lightgbm_quick|fp_morgan_desc|59.290|109.806|0.991|120.174|354.286|0.931|
|fp_morgan_desc_rf|fp_rf_quick|fp_morgan_desc|80.279|144.356|0.985|104.692|248.286|0.966|
|fp_morgan_desc_catboost|fp_catboost_quick|fp_morgan_desc|60.985|84.942|0.995|98.395|223.256|0.972|
|fp_morgan_desc_gpr|fp_gpr_quick|fp_morgan_desc|78.403|106.215|0.992|108.308|256.014|0.964|
|egnn_radius_quick|gnn_egnn_quick|gnn_radius_graph_quick|168.280|212.310|0.967|194.396|270.986|0.959|
|mpnn_radius_quick|gnn_mpnn_quick|gnn_radius_graph_quick|198.991|256.478|0.952|262.709|409.241|0.907|
|schnet_radius_quick|gnn_schnet_quick|gnn_radius_graph_quick|325.516|462.294|0.844|308.978|472.254|0.877|
|painn_radius_quick|gnn_painn_quick|gnn_radius_graph_quick|81.128|114.327|0.990|94.353|162.002|0.985|
|dimenetpp_radius_quick|dimenetpp|gnn_dimenetpp_quick|651.892|831.197|0.497|2644.424|28084.300|-435.629|
|gin_graph_quick|gin|gnn_graph_quick|165.863|215.326|0.966|213.724|331.398|0.939|
|graphormer_graph_quick|graphormer|gnn_graph_quick|78.163|97.281|0.993|87.529|143.153|0.989|
|graph_transformer_graph_quick|graph_transformer|gnn_graph_quick|78.163|97.281|0.993|87.529|143.153|0.989|
|smiles_chemberta|chemberta|smiles_tokenizer_qm9|786.392|978.622|0.302|836.973|1114.990|0.312|

Notes: Improved runs use target z-score transform; evaluation metrics are in original units after inverse transform. PaiNN runs via TorchMD-Net (equivariant-transformer) backend. DimeNet++ run uses `gnn_dimenetpp_quick` with `fallback_2d` conformer handling and shows unstable performance.

## Seed sweep (3 seeds, quick)
|model|seeds|val_mae (mean+/-std)|val_rmse (mean+/-std)|val_r2 (mean+/-std)|test_mae (mean+/-std)|test_rmse (mean+/-std)|test_r2 (mean+/-std)|
|---|---|---|---|---|---|---|---|
|egnn_radius_quick|41/42/43|128.795+/-7.483|166.978+/-8.653|0.980+/-0.002|152.904+/-19.633|226.004+/-25.680|0.971+/-0.006|
|painn_radius_quick|41/42/43|89.989+/-13.794|120.647+/-15.508|0.989+/-0.003|106.425+/-14.539|231.020+/-71.007|0.968+/-0.020|
|smiles_chemberta|41/42/43|760.621+/-23.394|949.032+/-21.826|0.343+/-0.030|822.504+/-11.563|1086.477+/-20.172|0.346+/-0.024|

Notes: Seed sweep uses `configs/gnn/train_qm9_quick_egnn.yaml`, `configs/gnn/train_qm9_quick_painn.yaml` (TorchMD-Net backend), and `configs/smiles/train_qm9_quick_improved.yaml` (with `freeze_encoder_epochs=1`).
