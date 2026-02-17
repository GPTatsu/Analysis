#!/bin/bash
# =============================================================
# 藤崎 メールマーケティング 週次レポート実行スクリプト
# =============================================================
#
# 【使い方】
#   bash run.sh 2026-02-02        ← 週開始日を指定して実行
#   bash run.sh                   ← 省略時は先週月曜日を自動計算
#
# 【事前準備】
#   data/YYYYMMDD/ フォルダに以下の3ファイルを配置：
#     - メール行動ログ.csv
#     - 会員マスタ.csv
#     - 受注データ.csv
#
# 【出力】
#   output/report_YYYYMMDD.html
# =============================================================

set -euo pipefail
cd "$(dirname "$0")"

WEEK_START="${1:-}"

if [ -n "$WEEK_START" ]; then
    python3 weekly_mail_report.py --week-start "$WEEK_START"
else
    python3 weekly_mail_report.py
fi
