# AI Content Screener Design (Current)

## 目的
- Google検索結果ページに、各検索結果の `Score` と `Judge` を表示する。
- 判定は拡張機能内で実行し、外部APIへ判定結果を送信しない。

## 現在の動作範囲
- 表示対象ページ: `https://www.google.com/*`, `https://www.google.co.jp/*`
- 表示対象パス: 実行時に `location.pathname === "/search"` を満たすページ
- 表示内容: `Score`（0.00-1.00）と `Judge`（Human / Unknown / AI）

## アーキテクチャ (MV3)
- `manifest.json`
  - `manifest_version: 3`
  - `permissions: ["storage"]`
  - `host_permissions: ["https://*/*"]`
  - `background.service_worker: src/background/service-worker.js`
  - `content_scripts`: `hash-model.js`, `hash-model-ja.js`, `main.js`
- `src/content/main.js`
  - Google検索結果DOMの走査
  - スニペット/取得HTMLから判定用テキストを生成
  - ローカル推論とカード描画
  - `chrome.storage.local` キャッシュ
- `src/background/service-worker.js`
  - 指定URLのHTML取得（`FETCH_HTML`）
  - タイムアウト/サイズ上限/HTML以外の除外
- `src/options/options.html`, `src/options/options.js`
  - 設定項目: `enabled`, `cacheTTLHours`

## 判定ロジック
- 文字3-gramハッシュ特徴のNaive Bayes系スコア
- モデル切替:
  - 既定モデル: `src/content/hash-model.js`
  - 日本語モデル: `src/content/hash-model-ja.js`
  - 日本語文字比率でモデルを自動選択
- 判定閾値:
  - `0.00 <= score < 0.45`: `Human`
  - `0.45 <= score < 0.55`: `Unknown`
  - `0.55 <= score <= 1.00`: `AI`

## テキスト抽出フロー
1. 検索結果カードのURLを抽出
2. Backgroundでリンク先HTMLを取得
3. `DOMParser` で解析し、`script/style/iframe/...` 等を除去
4. `article/main/[role=main]` など候補から本文量最大の要素を採用
5. 本文が短すぎる場合は、検索結果の `title + snippet` へフォールバック

## キャッシュ
- 保存先: `chrome.storage.local`
- キー: `cache:<url_hash>`
- 保存値: `score`, `judge`, `displayScore`, `source`, `updatedAt`, `cacheVersion`, `contentHash`
- TTL: 既定 `168` 時間（7日）

## UI仕様
- 結果カードは検索結果行の左側に配置
- 幅が狭い場合はインライン表示へフォールバック
- 表示文言は最小構成（`Score`, `Judge`）
- ライト/ダークを自動判定して配色切替

## 既知の制約
- 判定は推定であり、真偽を保証しない。
- データ分布が対象ドメインと異なると精度が低下しうる。
- ホスト権限が広いため、審査時に理由説明が必要。

## 公開リポジトリ運用ルール
- シークレット値（APIキー、トークン、個人情報）をコミットしない。
- 審査画面のスクリーンショットを公開する場合は個人情報をマスクする。
- 大容量データは `data/raw/`, `data/processed/` に置き、Git管理しない。
