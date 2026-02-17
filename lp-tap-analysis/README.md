# ショコラマルシェLP タップ分析

Clarity（モバイル）のLPタップデータを可視化するダッシュボードです。定常のメルマガ分析とは別の用途です。

## 使い方

- ブラウザで `index.html` を開く
- CDN（React / Recharts）を読み込むため、インターネット接続が必要です

## GitHub Pages でホストする場合

1. リポジトリに `lp-tap-analysis` フォルダを push
2. **Settings → Pages** で Source を「Deploy from a branch」に設定
3. ブランチとフォルダ（通常は `/ (root)`）を選択して保存
4. アクセスURL: `https://<username>.github.io/<repo>/lp-tap-analysis/`

## ファイル構成

```
lp-tap-analysis/
├── index.html   # 単体で動作するHTML（React + Recharts via CDN）
└── README.md
```
