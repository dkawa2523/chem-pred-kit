# Task 112 (P0): Radius Graph + Distance Edge Features

## 目的
SchNet等で使う `edge_attr`（距離/gaussian expansion）をFeatureSetとして生成する。

## In scope
- radius graph 構築（cutoff, max_neighbors をconfig化）
- distance 計算と gaussian basis 展開（optional）
- posが無い場合の明確なエラー（3Dモデルでは必須）

## Acceptance Criteria
- [x] 3D graph の edge_attr が生成され、モデルが forward できる
- [x] 生成パラメータが artifact/meta に残る
