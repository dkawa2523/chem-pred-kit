# Task 070 (P2): 事前学習埋め込み特徴量の導入（任意）

## 目的
- 1万件規模でも表現力を上げる手段として、公開済み分子埋め込みを利用可能にする

## Plan
1) featurizer として `pretrained_embedding` を追加
2) まずは “外部モデルを呼ばないスタブ” で I/F を確定
3) 次に HuggingFace などの実装を追加（依存追加は最小限）

## Acceptance Criteria
- [x] featureset で切替可能
- [x] 推論時も同じ埋め込みが再現できる
