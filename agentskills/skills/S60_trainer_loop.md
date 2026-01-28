# S60 Trainer Loop

## Purpose
学習ループの拡張（scheduler、early stopping、AMP等）を、設定で切替可能にする。

## Inputs
- docs/03_CONFIG_CONVENTIONS.md
- work/tasks/010 / 020 / 030 など

## Allowed Changes
- src/**/train*.py or src/core/**
- configs/train/**
- tests/**

## Common Pitfalls
- デフォルト挙動が変わって精度が落ちる
- 再現性（seed）を壊す
