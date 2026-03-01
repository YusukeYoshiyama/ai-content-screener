# Dataset Collection Log (Current)

## 目的
- ローカル判定器の評価・改善に使う `text` / `label` データを統合する。
- 日本語と英語の両方を含む評価セットを維持する。

## 使用データセット
- `gsingh1-py/train`
- `dmitva/human_ai_generated_text`
- `Aratako/Synthetic-Japanese-Roleplay-NSFW-gpt-5-chat-5k-formatted`
- `Aratako/Synthetic-Japanese-Roleplay-NSFW-Claude-4.5s-3.5k-formatted`
- `CausalLM/GPT-4-Self-Instruct-Japanese`
- `hpprc/jawiki-news-paragraphs`
- `hpprc/jawiki-books-paragraphs`

参照URLはHugging Face上の各データセットページ。

## 統合データ
- 出力: `data/processed/unified_text_label.parquet`
- カラム: `text`, `label`
- 行数: `2,268,493`
- ラベル内訳: `Human 1,209,962 / AI 1,058,531`

## モデル成果物
- 既定モデル: `data/processed/hash_nb_model_4096_sampled.json`
- 日本語モデル: `data/processed/hash_nb_model_4096_ja.json`
- 拡張同梱版:
  - `src/content/hash-model.js`
  - `src/content/hash-model-ja.js`

## 評価スクリプト
- `scripts/test_detector_on_dataset.py`
- `--workers` で並列実行可能
- `--model-ja` を指定すると日本語判定時に日本語モデルへ切替

実行例:
```bash
python3 scripts/test_detector_on_dataset.py \
  --input data/processed/unified_text_label.parquet \
  --model data/processed/hash_nb_model_4096_sampled.json \
  --model-ja data/processed/hash_nb_model_4096_ja.json \
  --workers 10 \
  --output data/processed/detector_eval_hybrid_full_default.json
```

## 公開リポジトリ注意事項
- `data/raw/`, `data/processed/` は `.gitignore` 対象。
- 実データや生成物はリポジトリに含めない（再現手順のみを記載）。
- データセットの利用規約・ライセンスは各配布元で確認する。
