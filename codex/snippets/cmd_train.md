# Train command snippets

例（実際の config 名に合わせて調整）:

```bash
# デフォルト設定で学習
python scripts/train.py

# モデルやタスクを切り替え
python scripts/train.py model.name=mpnn task.name=lj train.epochs=50
```
