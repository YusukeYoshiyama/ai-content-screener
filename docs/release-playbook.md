# Release Playbook (Current)

## ブランチ運用
- 開発: `develop`
- 本番反映: `develop` から `main` へPRを作成してマージ
- `main` への直接pushはしない

## CI/CD

### CI
- Workflow: `.github/workflows/ci.yml`
- Trigger:
  - `push` to `develop`
  - `pull_request` to `main`
  - `workflow_dispatch`
- 実行内容:
  - `manifest.json` のJSON検証
  - `src/**/*.js` の `node --check`
  - `scripts/*.py` の構文チェック
  - 拡張ZIP作成とartifact保存

### Release to Chrome Web Store
- Workflow: `.github/workflows/release-cws.yml`
- Trigger: `v*` タグpush
- 主な検証:
  - タグ形式 `vX.Y.Z`
  - タグコミットが `origin/main` に含まれる
  - タグのバージョンと `manifest.json` の `version` 一致

## 必要なSecrets (GitHub Actions)
- `CWS_CLIENT_ID`
- `CWS_CLIENT_SECRET`
- `CWS_REFRESH_TOKEN`
- `CWS_PUBLISHER_ID`
- `CWS_EXTENSION_ID`

## ローカルでZIPを作る
```bash
cd <repo-root>
scripts/build_extension_zip.sh
```

- 生成先: `dist/ai-content-screener-v<manifest.version>.zip`
- 追加チェック:
  - `manifest_version` が `3` であること
  - `manifest.icons` に定義したアイコン実ファイルが存在すること
  - ZIP内に `manifest.json` と各アイコンが含まれること

## リリース手順
1. `develop` でバージョン更新＋ZIP作成
```bash
git checkout develop
scripts/prepare_release.sh 0.1.1
git add manifest.json
git commit -m "chore: release v0.1.1"
git push origin develop
```
2. `develop -> main` のPRを作成しマージ
3. タグ作成とpush
```bash
git checkout main
git pull origin main
git tag v0.1.1
git push origin v0.1.1
```
4. Actionsの `Release to Chrome Web Store` 成功を確認

## 公開リポジトリ注意事項
- SecretsはGitHubの `Repository/Environment Secrets` のみで管理する。
- ローカル `.env` やトークン応答ファイルをコミットしない。
- 申請時の文面には、実装と一致する権限理由のみを記載する。
