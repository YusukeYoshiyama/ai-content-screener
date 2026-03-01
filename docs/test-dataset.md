# Evaluation Dataset Notes

## 位置づけ
このリポジトリでは、単一の500件サンプルよりも、統合データセット
`data/processed/unified_text_label.parquet` を主評価対象としている。

## 500件サンプルについて
- 生成スクリプト: `scripts/build_test_dataset.py`
- 出力:
  - `data/processed/articles_labeled_500.jsonl`
  - `data/processed/articles_labeled_500.csv`
- 用途:
  - 目視確認
  - UIデバッグ
  - スモークテスト

## 全件評価について
- 評価スクリプト: `scripts/test_detector_on_dataset.py`
- 推奨:
  - `--workers 10`
  - `--model-ja` 指定

例:
```bash
python3 scripts/test_detector_on_dataset.py \
  --input data/processed/unified_text_label.parquet \
  --model data/processed/hash_nb_model_4096_sampled.json \
  --model-ja data/processed/hash_nb_model_4096_ja.json \
  --workers 10 \
  --output data/processed/detector_eval_hybrid_full_default.json
```

## 判定境界
- `0.00-0.44`: Human
- `0.45-0.54`: Unknown
- `0.55-1.00`: AI

## 公開リポジトリ注意事項
- 評価結果JSONに個人情報や秘密情報を含めない。
- 共有時は統計値中心に記載し、生テキストの再配布可否を確認する。
