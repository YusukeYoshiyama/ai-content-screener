# Dataset Collection Log

## 目的
- AI生成判定の改善に向けて、500件を超える公開データセットを全件取得する
- 日本語データを追加して評価/調整に使える材料を増やす

## 取得したデータセット (全件)
- `gsingh1-py/train`
  - URL: https://huggingface.co/datasets/gsingh1-py/train
  - 行数: 7,321
  - 用途: `Human_story` vs 複数モデル出力の strong label
- `dmitva/human_ai_generated_text`
  - URL: https://huggingface.co/datasets/dmitva/human_ai_generated_text
  - 行数: 1,000,000
  - 用途: `human_text` / `ai_text` の strong label (英語)
- `Aratako/Synthetic-Japanese-Roleplay-NSFW-gpt-5-chat-5k-formatted`
  - URL: https://huggingface.co/datasets/Aratako/Synthetic-Japanese-Roleplay-NSFW-gpt-5-chat-5k-formatted
  - 行数: 5,019
  - 用途: 日本語AIテキスト (strong label)
- `Aratako/Synthetic-Japanese-Roleplay-NSFW-Claude-4.5s-3.5k-formatted`
  - URL: https://huggingface.co/datasets/Aratako/Synthetic-Japanese-Roleplay-NSFW-Claude-4.5s-3.5k-formatted
  - 行数: 3,482
  - 用途: 日本語AIテキスト (strong label)
- `CausalLM/GPT-4-Self-Instruct-Japanese`
  - URL: https://huggingface.co/datasets/CausalLM/GPT-4-Self-Instruct-Japanese
  - 行数: 6,144
  - 用途: 日本語AIテキスト (strong label)
- `hpprc/jawiki-news-paragraphs`
  - URL: https://huggingface.co/datasets/hpprc/jawiki-news-paragraphs
  - 行数: 16,633
  - 用途: 日本語Humanテキスト (weak label)
- `hpprc/jawiki-books-paragraphs`
  - URL: https://huggingface.co/datasets/hpprc/jawiki-books-paragraphs
  - 行数: 186,034
  - 用途: 日本語Humanテキスト (weak label)

## 保存先
- 生データ (parquet shards + manifest):
  - `data/raw/<dataset-idを__置換>/`

## 統合データセット (最終)
- `data/processed/unified_text_label.parquet`
  - カラム: `text`, `label`
  - 行数: `2,268,493`
  - ラベル内訳: `Human 1,209,962 / AI 1,058,531`
- サマリ:
  - `data/processed/unified_text_label_summary.json`

## スクリプト
- 取得:
  - `scripts/download_hf_parquet.py`
- 単一データセット生成 (`text`,`label`のみ):
  - `scripts/build_unified_text_label_dataset.py`
- 現在の判定ロジックで全件評価:
  - `scripts/test_detector_on_dataset.py`

## 再実行例
```bash
python3 scripts/download_hf_parquet.py --dataset gsingh1-py/train --out-dir data/raw
python3 scripts/download_hf_parquet.py --dataset dmitva/human_ai_generated_text --out-dir data/raw
python3 scripts/download_hf_parquet.py --dataset Aratako/Synthetic-Japanese-Roleplay-NSFW-gpt-5-chat-5k-formatted --out-dir data/raw
python3 scripts/download_hf_parquet.py --dataset Aratako/Synthetic-Japanese-Roleplay-NSFW-Claude-4.5s-3.5k-formatted --out-dir data/raw
python3 scripts/download_hf_parquet.py --dataset CausalLM/GPT-4-Self-Instruct-Japanese --out-dir data/raw
python3 scripts/download_hf_parquet.py --dataset hpprc/jawiki-news-paragraphs --out-dir data/raw
python3 scripts/download_hf_parquet.py --dataset hpprc/jawiki-books-paragraphs --out-dir data/raw

python3 scripts/build_unified_text_label_dataset.py --min-chars 1 --max-chars 8000 --batch-size 5000
python3 scripts/test_detector_on_dataset.py \
  --input data/processed/unified_text_label.parquet \
  --model data/processed/hash_nb_model_4096_sampled.json \
  --human-threshold 0.45 \
  --ai-threshold 0.55 \
  --max-rows 0 \
  --workers 10 \
  --output data/processed/detector_eval_summary.json
```

## 注意点
- 日本語の `Human vs AI` が同一設計で直接対応する公開データは少ないため、日本語は「Human側weak label」と「AI側strong label」の混成です。
- 最終評価には、あなたの対象ドメインに合わせた手動検証セットを追加するのが安全です。
