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

## Store Assets

Chrome Web Store の掲載情報は `store/` 配下で Git 管理します。

- `store/descriptions/`: 説明文
- `store/screenshots/`: スクリーンショット

反映自体は Developer Dashboard で手動対応します。
