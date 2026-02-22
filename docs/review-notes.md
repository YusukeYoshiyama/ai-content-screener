# Chrome Web Store Review Notes

## Current checklist
- [ ] 連絡先メール設定済み
- [ ] データ利用の表明を入力済み
- [ ] 単一用途の説明を入力済み
- [ ] `storage` 権限理由を入力済み
- [ ] host権限理由を入力済み
- [ ] リモートコード不使用の説明を入力済み

## 申請前チェック
- [ ] `manifest_version` が `3`
- [ ] ZIP再生成済み（`scripts/build_extension_zip.sh`）
- [ ] ZIP内にアイコン4種が含まれる
- [ ] 申請文面が現在の実装と一致

## メモ（公開リポジトリ運用）
- ストア申請画面の個人情報や内部URLはコミットしない。
- 審査の指摘内容は要約して記録し、機微情報は載せない。
