# ショコラマルシェLP タップ分析

Clarity（モバイル）のLPタップデータを可視化するダッシュボードです。定常のメルマガ分析とは別の用途です。

## 使い方

**重要:** `file://` で直接開くとCDNがブロックされ白画面になる場合があります。**ローカルサーバーで開いてください。**

```bash
cd lp-tap-analysis
python3 -m http.server 8080
```

ブラウザで http://localhost:8080/ を開く。インターネット接続が必要です。

## GitHub Pages でホストする場合

1. リポジトリに `lp-tap-analysis` フォルダを push
2. **Settings → Pages** で Source を「Deploy from a branch」に設定
3. ブランチとフォルダ（通常は `/ (root)`）を選択して保存
4. アクセスURL: `https://gptatsu.github.io/Analysis/lp-tap-analysis/`

## ファイル構成

```
lp-tap-analysis/
├── index.html   # 単体で動作するHTML（React + Recharts via CDN）
└── README.md
```
