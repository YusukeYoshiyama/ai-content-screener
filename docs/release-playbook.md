# Release Playbook

## Branch strategy

- 通常開発は `develop` で行う
- `main` へのマージは Pull Request 経由のみ
- `main` へのPR時に CI を実行して品質確認

## Required secrets (GitHub Actions)

- `CWS_CLIENT_ID`
- `CWS_CLIENT_SECRET`
- `CWS_REFRESH_TOKEN`
- `CWS_PUBLISHER_ID`
- `CWS_EXTENSION_ID`

## CI flow

- トリガー:
  - `pull_request` to `main`
  - `push` to `develop`
- 実行内容:
  - `manifest.json` のJSON検証
  - `src/**/*.js` の `node --check`
  - `scripts/*.py` の構文検証
  - 配布用zip作成とartifact保存

## Release flow

1. `develop` -> `main` のPRをマージ
2. `manifest.json` の `version` を確認
3. タグを作成してpush

```bash
git checkout main
git pull origin main
git tag v0.1.0
git push origin v0.1.0
```

4. GitHub Actions の `Release to Chrome Web Store` を確認
5. `production` environment 承認が必要な場合は承認
6. publish成功ログを確認

## Important notes

- タグ名 (`vX.Y.Z`) と `manifest.json` の `version` は一致必須
- 不一致の場合、release workflow は失敗する仕様
- 公開失敗時は `upload_response.json` と `publish_response.json` のログを確認する

