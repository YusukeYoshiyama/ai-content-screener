# AI Content Screener

Google 検索結果ページで、各検索結果に `Score` と `Judge` を表示する Chrome 拡張です。

- 対象ページ: `https://www.google.com/*`, `https://www.google.co.jp/*`
- 判定結果: `Human`, `Unknown`, `AI`
- 判定処理: ローカル実行

## Project Structure

- `src/`: 拡張本体
- `icon/`: 拡張アイコン
- `store/`: Chrome Web Store 掲載用の説明文と画像
- `scripts/release.sh`: ZIP 作成とリリース準備
- `scripts/data_tools.py`: データ取得、データ生成、評価

## Local Development

1. Chrome の拡張機能管理画面でデベロッパーモードを有効化
2. `パッケージ化されていない拡張機能を読み込む` でこのリポジトリを指定
3. Google 検索結果ページで動作確認

## Release

ZIP 作成:

```bash
scripts/release.sh build-zip
```

リリース準備:

```bash
scripts/release.sh prepare 0.1.1
```

`vX.Y.Z` タグを `main` に push すると、GitHub Actions から Chrome Web Store 公開フローが走ります。

## Data Tools

Hugging Face データ取得:

```bash
python3 scripts/data_tools.py download-hf --dataset dmitva/human_ai_generated_text
```

統合データセット作成:

```bash
python3 scripts/data_tools.py build-unified
```

評価実行:

```bash
python3 scripts/data_tools.py evaluate --model data/processed/hash_nb_model_4096_sampled.json --model-ja data/processed/hash_nb_model_4096_ja.json
```

ライブURLの検証:

```bash
python3 scripts/data_tools.py verify-live --input data/live_eval/web_human.csv --write-manifest
```

ライブURLマニフェスト生成:

```bash
python3 scripts/data_tools.py collect-live-seed --spec /tmp/live_specs.json --output data/live_eval/web_human.csv
```

ライブURL評価:

```bash
python3 scripts/data_tools.py evaluate-live --input data/live_eval/web_human.csv --model data/processed/hybrid_hash_nb_model_4096_sampled.json --model-ja data/processed/hybrid_hash_nb_model_4096_ja.json
```

live suite 評価:

```bash
python3 scripts/data_tools.py evaluate-live-suite --model data/processed/hybrid_hash_nb_model_4096_sampled.json --model-ja data/processed/hybrid_hash_nb_model_4096_ja.json
```

軽量ハイブリッドモデル再学習:

```bash
python3 scripts/data_tools.py train-hybrid --live-human-manifest data/live_eval/web_human.csv --live-ai-page-manifest data/live_eval/web_ai_page.csv --live-ai-site-manifest data/live_eval/web_ai_site.csv --serp-audit-manifest data/live_eval/serp_audit.csv --regression-manifest data/live_eval/regression_cases.csv
```

## Store Assets

Chrome Web Store の掲載情報は `store/` 配下で Git 管理します。

- `store/descriptions/`: 説明文
- `store/screenshots/`: スクリーンショット

反映自体は Developer Dashboard で手動対応します。

## Live Eval

実サイト評価用の URL マニフェストは `data/live_eval/` 配下で管理します。

- `web_human.csv`: `400` 件の Human 回帰セット
  - うち `80` 件は AI関連の hard negative
- `web_ai_page.csv`: `200` 件の page-level 明示 AI セット
- `web_ai_site.csv`: `100` 件の site-level AI セット
- `serp_audit.csv`: `30` クエリ x `10` 件、計 `300` 件の検索監査セット
- `regression_cases.csv`: 既知の誤判定回帰ケース

取得した HTML や本文キャッシュは `data/live_cache/` に保存し、Git には含めません。
