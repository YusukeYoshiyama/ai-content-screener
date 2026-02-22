# Test Dataset (500 samples)

## 概要
- 目的: AI記事判定ロジックの閾値調整・評価用データを用意する
- 出力:
  - `data/processed/articles_labeled_500.jsonl`
  - `data/processed/articles_labeled_500.csv`
- 件数: 500件 (`Human` 250 / `AI` 250)

## データソース
- `gsingh1-py/train` (Hugging Face)
- 入力ファイル:
  - `data/raw/gsingh_train.parquet`

## ラベル付けルール
- `Human`: `Human_story`列
- `AI`: モデル出力列
  - `gemma-2-9b`
  - `mistral-7B`
  - `qwen-2-72B`
  - `llama-8B`
  - `accounts/yi-01-ai/models/yi-large`
  - `GPT_4-o`
- `label_confidence`: 一律 `0.98` (データセット提供ラベル準拠)

## 生成コマンド
```bash
python3 scripts/build_test_dataset.py --size 500 --seed 42 --min-chars 500 --max-chars 8000
```

## メタデータ項目
- `id`
- `label` (`AI` / `Human`)
- `label_confidence`
- `label_reason`
- `source_dataset`
- `source_row`
- `prompt`
- `model`
- `original_text_length`
- `text_length`
- `text_hash`
- `retrieved_at`
- `text`

## 注意点
- 本データは「WebページURL単位」ではなく、公開コーパスのテキスト単位データです。
- 検索結果ページのUI検証には、別途URL付きデータを併用してください。
- `text`は評価しやすい粒度に合わせて最大`8000`文字へ切り詰めています。
