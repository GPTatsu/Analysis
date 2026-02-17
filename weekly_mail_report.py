#!/usr/bin/env python3
"""
==============================================================================
藤崎メールマーケティング 週次分析レポート生成スクリプト v3
==============================================================================

【v3 変更点】
  - 施策を3タイプに自動分類（EC購買 / CRM / 公式サイト流入）
  - タイプ別CVR定義（受注CVR / 開封CVR / クリックCVR）
  - アトリビューションウィンドウ: KPIは3日以内、テーブルで同日/3日/7日を並列表示
  - 許諾率の過去4週トレンドを表示
  - history.csv に年代別許諾率を追記

【フォルダ構成】
  Analysis/
  ├── data/
  │   ├── shared/                  ← 四半期ログ等、複数週共通のCSVを配置
  │   └── YYYYMMDD/                ← 週開始日(月曜)ごとのフォルダ
  │       ├── メール行動ログ.csv
  │       ├── 会員マスタ.csv
  │       └── 受注データ.csv
  ├── output/
  │   └── report_YYYYMMDD.html
  ├── history.csv
  ├── weekly_mail_report.py
  └── run.sh

【使い方】
  python weekly_mail_report.py --week-start 2026-02-02
  bash run.sh 2026-02-02

【CVRアトリビューションルール】
  顧客ID一致 ＋ メール行動日から N日以内の受注 → メール経由受注とカウント
  KPI表示: 3日以内 / テーブル: 同日・3日以内・7日以内を並列表示
==============================================================================
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ====================================================================
# 0. ユーティリティ
# ====================================================================
def pct(num, den):
    if den == 0:
        return 0.0
    return round(num / den * 100, 2)


def age_bucket(age):
    if pd.isna(age) or age < 0:
        return "不明"
    if age < 20:   return "10代以下"
    elif age < 30: return "20代"
    elif age < 40: return "30代"
    elif age < 50: return "40代"
    elif age < 60: return "50代"
    elif age < 70: return "60代"
    else:          return "70代以上"


AGE_ORDER = ["10代以下", "20代", "30代", "40代", "50代", "60代", "70代以上", "不明"]


def fmt_delta(current, previous, suffix="pt", fmt=".1f", is_money=False):
    if previous is None or pd.isna(previous):
        return "—"
    diff = current - previous
    arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
    if is_money:
        return f"{arrow}¥{abs(int(diff)):,}"
    return f"{arrow}{abs(diff):{fmt}}{suffix}"


def delta_color(current, previous, higher_is_better=True):
    if previous is None or pd.isna(previous):
        return "#888"
    diff = current - previous
    if diff == 0:
        return "#888"
    if higher_is_better:
        return "#16a34a" if diff > 0 else "#dc2626"
    else:
        return "#dc2626" if diff > 0 else "#16a34a"


# ====================================================================
# 0b. 施策3分類
# ====================================================================
CAMPAIGN_TYPES = {
    "EC購買": "EC購買",
    "CRM": "CRM",
    "公式サイト流入": "公式サイト流入",
}

CAMPAIGN_TYPE_ORDER = ["EC購買", "CRM", "公式サイト流入"]

# CVR定義: タイプ別
CVR_DEFINITIONS = {
    "EC購買":       {"label": "受注CVR",     "formula": "受注数 ÷ クリック数",    "num": "受注", "den": "クリック"},
    "CRM":          {"label": "開封CVR",     "formula": "開封数 ÷ 配信成功数",    "num": "開封", "den": "配信成功"},
    "公式サイト流入": {"label": "クリックCVR", "formula": "クリック数 ÷ 開封数",    "num": "クリック", "den": "開封"},
}


def classify_campaign(name):
    """施策名からタイプを判定。CRM優先 → EC購買 → 公式サイト流入"""
    n = str(name)
    # CRM優先
    if "【CRM】" in n or "CRM" in n:
        return "CRM"
    if "化粧品サンクス" in n or "化粧品購入後" in n:
        return "CRM"
    if "リマインド" in n and "【CRM】" not in n:
        # リマインド系もCRM
        if "リマインド" in n:
            return "CRM"
    if "友の会" in n and "自動加算" in n:
        return "CRM"
    # EC購買
    if "ECメルマガ" in n or "ＥＣメルマガ" in n:
        return "EC購買"
    if "かご落ち" in n:
        return "EC購買"
    if "ポイント・クーポン等付与" in n or "ポイントプレゼント" in n:
        return "EC購買"
    if "クーポン" in n:
        return "EC購買"
    if "ログインキャンペーン" in n:
        return "EC購買"
    # 公式サイト流入
    return "公式サイト流入"


# ====================================================================
# 1. データ読み込み・前処理
# ====================================================================
def load_data(log_path, members_path, orders_path, encoding="cp932",
              shared_dir=None, members_optional=False, orders_optional=False):
    print(f"[1/6] データ読み込み中...")

    # --- メール行動ログ ---
    if log_path == "__shared_combined__" and shared_dir:
        shared_csvs = sorted(Path(shared_dir).glob("*.csv"))
        dfs = []
        for csv_path in shared_csvs:
            df_part = pd.read_csv(str(csv_path), encoding=encoding, dtype={"顧客ID": str})
            dfs.append(df_part)
            print(f"  shared: {csv_path.name} ({len(df_part):,} 行)")
        df_log = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["メール行動ログID"])
    else:
        df_log = pd.read_csv(log_path, encoding=encoding, dtype={"顧客ID": str})
    df_log["行動日時"] = pd.to_datetime(df_log["行動日時"], errors="coerce")
    df_log["配信日時"] = pd.to_datetime(df_log["配信日時"], errors="coerce")
    df_log["配信日"] = df_log["配信日時"].dt.date
    df_log["行動日"] = df_log["行動日時"].dt.date

    # 施策タイプ分類
    df_log["施策タイプ"] = df_log["施策名"].apply(classify_campaign)

    print(f"  メール行動ログ（結合後）: {len(df_log):,} 行")

    # --- 会員マスタ ---
    if members_optional:
        df_mem = pd.DataFrame(columns=["EC会員番号", "会員ステータス", "メール受信許諾", "年齢", "年代", "性別"])
        print(f"  会員マスタ: なし（スキップ）")
    else:
        df_mem = pd.read_csv(members_path, encoding=encoding, dtype={"EC会員番号": str})

    if "会員状態" in df_mem.columns:
        df_mem["会員ステータス"] = df_mem["会員状態"].apply(
            lambda x: "有効" if str(x).startswith("0") else "退会"
        )

    if "案内メール希望" in df_mem.columns:
        df_mem["メール受信許諾"] = df_mem["案内メール希望"].apply(
            lambda x: 1 if str(x).startswith("1") or str(x).startswith("3") else 0
        )

    if "生年月日" in df_mem.columns:
        df_mem["生年月日_dt"] = pd.to_datetime(df_mem["生年月日"], errors="coerce")
        today = pd.Timestamp.now()
        age_days = (today - df_mem["生年月日_dt"]).dt.days
        df_mem["年齢_calc"] = pd.to_numeric(age_days / 365.25, errors="coerce").apply(
            lambda x: int(x) if pd.notna(x) else None
        )
    if "年齢" not in df_mem.columns and "年齢_calc" in df_mem.columns:
        df_mem["年齢"] = df_mem["年齢_calc"]
    if "年齢" in df_mem.columns:
        df_mem["年代"] = df_mem["年齢"].apply(age_bucket)

    if "性別" in df_mem.columns:
        df_mem["性別"] = df_mem["性別"].apply(
            lambda x: str(x).split("：")[-1] if pd.notna(x) and "：" in str(x) else x
        )

    print(f"  会員マスタ: {len(df_mem):,} 行")

    # --- 受注データ ---
    if orders_optional:
        df_ord = pd.DataFrame(columns=["会員SEQ", "受注日時", "受注日", "受注金額"])
        print(f"  受注データ: なし（スキップ）")
    else:
        df_ord = pd.read_csv(orders_path, encoding=encoding, dtype={"会員SEQ": str})
        df_ord["受注日時"] = pd.to_datetime(df_ord["受注日時"], errors="coerce")
        df_ord["受注日"] = df_ord["受注日時"].dt.date
        print(f"  受注データ: {len(df_ord):,} 行")

    # --- ログに会員情報をJOIN ---
    join_cols = ["EC会員番号", "性別", "年齢", "年代", "メール受信許諾"]
    join_cols = [c for c in join_cols if c in df_mem.columns]
    if len(join_cols) > 0:
        df_log = df_log.merge(
            df_mem[join_cols],
            left_on="顧客ID", right_on="EC会員番号", how="left"
        )

    return df_log, df_mem, df_ord


# ====================================================================
# 2. 対象週のフィルタリング
# ====================================================================
def filter_week(df_log, df_ord, week_start):
    ws = pd.Timestamp(week_start)
    we = ws + timedelta(days=6, hours=23, minutes=59, seconds=59)
    print(f"[2/6] 対象期間: {ws.strftime('%Y-%m-%d')} 〜 {we.strftime('%Y-%m-%d')}")

    log_w = df_log[(df_log["配信日時"] >= ws) & (df_log["配信日時"] <= we)].copy()

    # 受注データは、ウィンドウ7日を考慮して広めに取る（配信週+7日後まで）
    ord_end = we + timedelta(days=7)
    ord_w = df_ord[(df_ord["受注日時"] >= ws) & (df_ord["受注日時"] <= ord_end)].copy()

    print(f"  対象ログ: {len(log_w):,} 行 / 対象受注: {len(ord_w):,} 行")
    return log_w, ord_w


# ====================================================================
# 3. 集計エンジン
# ====================================================================
def _compute_orders_with_window(grp, ord_w, window_days):
    """指定ウィンドウ日数で受注をマッチング"""
    clicked = grp[grp["行動タイプ"] == "mail_clicked"]
    if len(clicked) == 0 or len(ord_w) == 0:
        return 0, 0

    click_data = clicked[["顧客ID", "行動日"]].drop_duplicates()
    order_data = ord_w[["会員SEQ", "受注日", "受注金額"]].copy()
    order_data = order_data.rename(columns={"会員SEQ": "顧客ID"})

    merged = click_data.merge(order_data, on="顧客ID", how="inner")
    merged["日数差"] = (pd.to_datetime(merged["受注日"]) - pd.to_datetime(merged["行動日"])).dt.days
    matched = merged[(merged["日数差"] >= 0) & (merged["日数差"] <= window_days)]

    n_customers = matched["顧客ID"].nunique()
    total_amount = int(matched["受注金額"].sum()) if "受注金額" in matched.columns and len(matched) > 0 else 0
    return n_customers, total_amount


def compute_funnel(log_w, ord_w, group_cols=None):
    if group_cols is None:
        group_cols = []

    results = []
    if group_cols:
        groups = log_w.groupby(group_cols)
    else:
        groups = [("全体", log_w)]

    for group_key, grp in groups:
        row = {}
        if group_cols:
            if isinstance(group_key, tuple):
                for col, val in zip(group_cols, group_key):
                    row[col] = val
            else:
                row[group_cols[0]] = group_key

        # 基本指標
        for action_type, label in [("mail_tried", "配信試行"), ("mail_failed", "配信失敗"),
                                    ("mail_opened", "開封"), ("mail_clicked", "クリック")]:
            subset = grp[grp["行動タイプ"] == action_type]
            row[f"{label}_人数"] = subset["顧客ID"].nunique()

        tried_ids = set(grp[grp["行動タイプ"] == "mail_tried"]["顧客ID"].dropna().unique())
        failed_ids = set(grp[grp["行動タイプ"] == "mail_failed"]["顧客ID"].dropna().unique())
        row["配信成功_人数"] = len(tried_ids - failed_ids)

        # 受注: 3つのウィンドウで算出
        for window, suffix in [(0, "_同日"), (3, "_3日"), (7, "_7日")]:
            n_cust, amount = _compute_orders_with_window(grp, ord_w, window)
            row[f"受注_人数{suffix}"] = n_cust
            row[f"受注金額{suffix}"] = amount

        # KPI用は3日以内
        row["受注_人数"] = row["受注_人数_3日"]
        row["受注金額"] = row["受注金額_3日"]

        row["開封率"] = pct(row["開封_人数"], row["配信成功_人数"])
        row["クリック率"] = pct(row["クリック_人数"], row["開封_人数"])

        # 受注CVR (3日以内)
        row["受注CVR"] = pct(row["受注_人数"], row["クリック_人数"])
        row["受注CVR_同日"] = pct(row["受注_人数_同日"], row["クリック_人数"])
        row["受注CVR_3日"] = pct(row["受注_人数_3日"], row["クリック_人数"])
        row["受注CVR_7日"] = pct(row["受注_人数_7日"], row["クリック_人数"])

        # 開封CVR = 開封 / 配信成功 (= 開封率と同じ)
        row["開封CVR"] = row["開封率"]

        # クリックCVR = クリック / 開封 (= クリック率と同じ)
        row["クリックCVR"] = row["クリック率"]

        row["バウンス率"] = pct(row["配信失敗_人数"], row["配信試行_人数"])
        row["配信停止率"] = 0.0

        # 施策タイプ別CVR
        if "施策タイプ" in row:
            ctype = row["施策タイプ"]
        elif "施策名" in row:
            ctype = classify_campaign(row["施策名"])
        else:
            ctype = "EC購買"
        row["施策タイプ_cvr"] = ctype
        cvr_def = CVR_DEFINITIONS.get(ctype, CVR_DEFINITIONS["EC購買"])
        if cvr_def["num"] == "受注":
            row["目標CVR"] = row["受注CVR"]
        elif cvr_def["num"] == "開封":
            row["目標CVR"] = row["開封CVR"]
        else:
            row["目標CVR"] = row["クリックCVR"]

        results.append(row)

    return pd.DataFrame(results)


def compute_funnel_by_type(log_w, ord_w):
    """施策タイプ別の集計"""
    return compute_funnel(log_w, ord_w, group_cols=["施策タイプ"])


def compute_subject_ranking(log_w):
    tried = log_w[log_w["行動タイプ"] == "mail_tried"].groupby("メール件名")["顧客ID"].nunique().reset_index()
    tried.columns = ["メール件名", "配信試行数"]
    failed = log_w[log_w["行動タイプ"] == "mail_failed"].groupby("メール件名")["顧客ID"].nunique().reset_index()
    failed.columns = ["メール件名", "配信失敗数"]
    delivered = tried.merge(failed, on="メール件名", how="left").fillna(0)
    delivered["配信失敗数"] = delivered["配信失敗数"].astype(int)
    delivered["配信成功数"] = delivered["配信試行数"] - delivered["配信失敗数"]

    opened = log_w[log_w["行動タイプ"] == "mail_opened"].groupby("メール件名")["顧客ID"].nunique().reset_index()
    opened.columns = ["メール件名", "開封者数"]
    df = delivered.merge(opened, on="メール件名", how="left").fillna(0)
    df["開封者数"] = df["開封者数"].astype(int)
    df["開封率"] = df.apply(lambda r: pct(r["開封者数"], r["配信成功数"]), axis=1)
    df = df[["メール件名", "配信成功数", "開封者数", "開封率"]]
    df = df.sort_values("開封率", ascending=False).reset_index(drop=True)
    return df


def compute_hourly_heatmap(log_w):
    log_w = log_w.copy()
    log_w["配信時間帯"] = log_w["配信日時"].dt.hour
    log_w["配信曜日"] = log_w["配信日時"].dt.day_name()

    tried = log_w[log_w["行動タイプ"] == "mail_tried"].groupby(["配信曜日", "配信時間帯"])["顧客ID"].nunique().reset_index()
    tried.columns = ["曜日", "時間帯", "配信試行数"]
    failed = log_w[log_w["行動タイプ"] == "mail_failed"].groupby(["配信曜日", "配信時間帯"])["顧客ID"].nunique().reset_index()
    failed.columns = ["曜日", "時間帯", "配信失敗数"]
    delivered = tried.merge(failed, on=["曜日", "時間帯"], how="left").fillna(0)
    delivered["配信失敗数"] = delivered["配信失敗数"].astype(int)
    delivered["配信成功数"] = delivered["配信試行数"] - delivered["配信失敗数"]

    opened = log_w[log_w["行動タイプ"] == "mail_opened"].groupby(["配信曜日", "配信時間帯"])["顧客ID"].nunique().reset_index()
    opened.columns = ["曜日", "時間帯", "開封者数"]
    df = delivered.merge(opened, on=["曜日", "時間帯"], how="left").fillna(0)
    df["開封者数"] = df["開封者数"].astype(int)
    df["開封率"] = df.apply(lambda r: pct(r["開封者数"], r["配信成功数"]), axis=1)
    df = df[["曜日", "時間帯", "配信成功数", "開封者数", "開封率"]]
    return df


def compute_optin_health(df_mem):
    if len(df_mem) == 0 or "会員ステータス" not in df_mem.columns:
        return pd.DataFrame(columns=["年代", "会員数", "オプトイン数", "オプトイン率"])
    optin = df_mem[df_mem["会員ステータス"] == "有効"].groupby("年代").agg(
        会員数=("EC会員番号", "count"),
        オプトイン数=("メール受信許諾", "sum")
    ).reset_index()
    optin["オプトイン率"] = optin.apply(lambda r: pct(r["オプトイン数"], r["会員数"]), axis=1)
    optin["_sort"] = optin["年代"].apply(lambda x: AGE_ORDER.index(x) if x in AGE_ORDER else 99)
    optin = optin.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)
    return optin


def compute_optin_total(df_mem):
    """全体の許諾率を算出"""
    if len(df_mem) == 0 or "会員ステータス" not in df_mem.columns:
        return 0.0
    active = df_mem[df_mem["会員ステータス"] == "有効"]
    if len(active) == 0:
        return 0.0
    return pct(active["メール受信許諾"].sum(), len(active))


# ====================================================================
# 4. 履歴管理（history.csv）
# ====================================================================
HISTORY_COLS = [
    "week_start", "配信成功_人数", "開封_人数", "開封率",
    "クリック_人数", "クリック率", "受注_人数", "受注CVR",
    "受注金額", "バウンス率", "配信停止率",
    "許諾率_全体",
    "許諾率_20代", "許諾率_30代", "許諾率_40代",
    "許諾率_50代", "許諾率_60代", "許諾率_70代以上",
]


def load_history(history_path):
    if os.path.exists(history_path):
        df = pd.read_csv(history_path, encoding="utf-8-sig")
        df["week_start"] = pd.to_datetime(df["week_start"]).dt.strftime("%Y-%m-%d")
        return df
    return pd.DataFrame(columns=HISTORY_COLS)


def save_history(history_path, history_df, summary, week_start_str,
                 optin_health=None, optin_total=0.0):
    row = {"week_start": week_start_str}
    if len(summary) > 0:
        s = summary.iloc[0]
        for col in HISTORY_COLS:
            if col == "week_start":
                continue
            if col.startswith("許諾率_"):
                continue
            row[col] = s.get(col, 0)

    row["許諾率_全体"] = optin_total
    if optin_health is not None and len(optin_health) > 0:
        for _, r in optin_health.iterrows():
            age = r["年代"]
            col_name = f"許諾率_{age}"
            if col_name in HISTORY_COLS:
                row[col_name] = r["オプトイン率"]

    new_row = pd.DataFrame([row])
    if week_start_str in history_df["week_start"].values:
        history_df = history_df[history_df["week_start"] != week_start_str]

    history_df = pd.concat([history_df, new_row], ignore_index=True)
    history_df = history_df.sort_values("week_start").reset_index(drop=True)
    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")
    print(f"  履歴を保存: {history_path} ({len(history_df)} 週分)")
    return history_df


def get_comparison(history_df, week_start_str):
    """前週・過去4週平均・前年同週のデータを取得"""
    ws = pd.Timestamp(week_start_str)

    prev_ws = (ws - timedelta(days=7)).strftime("%Y-%m-%d")
    prev_row = history_df[history_df["week_start"] == prev_ws]
    prev = prev_row.iloc[0].to_dict() if len(prev_row) > 0 else None

    past_weeks = history_df[history_df["week_start"] < week_start_str].tail(4)
    if len(past_weeks) > 0:
        avg = {}
        for col in HISTORY_COLS:
            if col != "week_start":
                avg[col] = past_weeks[col].mean()
        avg["_weeks"] = len(past_weeks)
    else:
        avg = None

    yoy_ws = (ws - timedelta(weeks=52)).strftime("%Y-%m-%d")
    yoy_row = history_df[history_df["week_start"] == yoy_ws]
    if len(yoy_row) == 0:
        yoy_ws_m1 = (ws - timedelta(weeks=52) - timedelta(days=7)).strftime("%Y-%m-%d")
        yoy_ws_p1 = (ws - timedelta(weeks=52) + timedelta(days=7)).strftime("%Y-%m-%d")
        yoy_row = history_df[history_df["week_start"].isin([yoy_ws_m1, yoy_ws_p1])]
    yoy = yoy_row.iloc[0].to_dict() if len(yoy_row) > 0 else None

    return prev, avg, yoy


def get_optin_trend(history_df, week_start_str, n_weeks=4):
    """過去N週の許諾率トレンドを取得（会員マスタありの週のみ）"""
    past = history_df[history_df["week_start"] <= week_start_str].copy()
    # 許諾率_全体が0より大きい週のみ（会員マスタがある週）
    past = past[past["許諾率_全体"] > 0]
    return past.tail(n_weeks + 1)


# ====================================================================
# 5. HTMLレポート生成
# ====================================================================
CSS = """
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: "Hiragino Kaku Gothic ProN", "Yu Gothic", "Meiryo", sans-serif; background: #f5f5f5; color: #333; line-height: 1.6; padding: 24px; }
    .container { max-width: 1200px; margin: 0 auto; }
    h1 { font-size: 20px; color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 8px; margin-bottom: 24px; }
    h2 { font-size: 16px; color: #1a1a2e; margin: 32px 0 12px; padding-left: 12px; border-left: 4px solid #e94560; }
    h3 { font-size: 14px; color: #555; margin: 16px 0 8px; }
    .meta { color: #888; font-size: 13px; margin-bottom: 24px; }
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }
    .kpi-card { background: #fff; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); text-align: center; }
    .kpi-card .label { font-size: 12px; color: #888; margin-bottom: 4px; }
    .kpi-card .value { font-size: 24px; font-weight: bold; color: #1a1a2e; }
    .kpi-card .unit { font-size: 12px; color: #888; }
    .kpi-card.accent .value { color: #e94560; }
    table { width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 16px; font-size: 13px; }
    th { background: #1a1a2e; color: #fff; padding: 10px 12px; text-align: left; font-weight: 500; white-space: nowrap; }
    th.sub { background: #2d2d4e; font-size: 11px; }
    td { padding: 8px 12px; border-bottom: 1px solid #eee; }
    tr:hover { background: #f9f9f9; }
    .highlight { font-weight: bold; color: #e94560; }
    .section { margin-bottom: 32px; }
    .note { background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px 16px; border-radius: 4px; font-size: 13px; margin: 16px 0; }
    .note-info { background: #e8f4fd; border-left: 4px solid #3b82f6; padding: 12px 16px; border-radius: 4px; font-size: 13px; margin: 16px 0; }
    .bar-container { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
    .bar { height: 16px; border-radius: 4px; min-width: 2px; }
    .bar-label { font-size: 12px; white-space: nowrap; }
    .type-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; color: #fff; margin-right: 4px; }
    .type-ec { background: #e94560; }
    .type-crm { background: #3b82f6; }
    .type-site { background: #10b981; }
    .trend-table td { text-align: center; }
    .trend-up { color: #16a34a; }
    .trend-down { color: #dc2626; }
    @media print { body { padding: 0; background: #fff; } .kpi-card { box-shadow: none; border: 1px solid #ddd; } table { box-shadow: none; border: 1px solid #ddd; } }
</style>
"""


def kpi_card_html(label, value, unit, prev_val=None, avg_val=None, yoy_val=None,
                  accent=False, higher_is_better=True, is_pct=True):
    accent_cls = ' accent' if accent else ''
    is_money = "金額" in label
    is_count = not is_pct and not is_money
    if is_pct:
        val_display = f"{value:.1f}%"
    elif is_money:
        val_display = f"&yen;{int(value):,}"
    else:
        val_display = f"{int(value):,}"

    suffix = "pt" if is_pct else ""

    if is_count:
        suffix = ""
        fmt_str = ".0f"
    else:
        fmt_str = ".1f"

    delta_html = ""
    if prev_val is not None and not pd.isna(prev_val):
        d = fmt_delta(value, prev_val, suffix=suffix, fmt=fmt_str, is_money=is_money)
        c = delta_color(value, prev_val, higher_is_better)
        delta_html += f'<div style="font-size:12px;font-weight:600;color:{c};margin-top:4px;">前週比 {d}</div>'

    if avg_val is not None and not pd.isna(avg_val):
        d = fmt_delta(value, avg_val, suffix=suffix, fmt=fmt_str, is_money=is_money)
        c = delta_color(value, avg_val, higher_is_better)
        delta_html += f'<div style="font-size:11px;color:{c};">平均比 {d}</div>'

    if yoy_val is not None and not pd.isna(yoy_val):
        d = fmt_delta(value, yoy_val, suffix=suffix, fmt=fmt_str, is_money=is_money)
        c = delta_color(value, yoy_val, higher_is_better)
        delta_html += f'<div style="font-size:11px;color:{c};">前年同週比 {d}</div>'

    return f"""    <div class="kpi-card{accent_cls}">
        <div class="label">{label}</div>
        <div class="value">{val_display}</div>
        <div class="unit">{unit}</div>
        {delta_html}
    </div>
    """


def _type_badge(ctype):
    cls_map = {"EC購買": "type-ec", "CRM": "type-crm", "公式サイト流入": "type-site"}
    return f'<span class="type-badge {cls_map.get(ctype, "type-site")}">{ctype}</span>'


def _cvr_label(ctype):
    return CVR_DEFINITIONS.get(ctype, CVR_DEFINITIONS["EC購買"])["label"]


def generate_key_insights(summary, by_type, by_campaign, by_age, optin_health,
                          subject_rank, prev, avg):
    """数値データからインサイトを自動生成"""
    insights = []
    if len(summary) == 0:
        return insights

    s = summary.iloc[0]
    prev_d = prev if prev else {}
    avg_d = avg if avg else {}

    # --- 1. 全体パフォーマンス概況 ---
    open_rate = s.get("開封率", 0)
    click_rate = s.get("クリック率", 0)
    cvr = s.get("受注CVR", 0)
    revenue = int(s.get("受注金額", 0))
    delivered = int(s.get("配信成功_人数", 0))

    prev_open = prev_d.get("開封率")
    prev_click = prev_d.get("クリック率")
    prev_cvr = prev_d.get("受注CVR")
    prev_revenue = prev_d.get("受注金額")

    # 全体トレンド判定
    declines = []
    improvements = []
    if prev_open is not None and not pd.isna(prev_open):
        d = open_rate - prev_open
        if d <= -3:
            declines.append(f"開封率が前週比 {d:+.1f}pt（{open_rate:.1f}%）と大幅に低下")
        elif d <= -1:
            declines.append(f"開封率が前週比 {d:+.1f}pt（{open_rate:.1f}%）とやや低下")
        elif d >= 3:
            improvements.append(f"開封率が前週比 {d:+.1f}pt（{open_rate:.1f}%）と大幅に改善")
        elif d >= 1:
            improvements.append(f"開封率が前週比 {d:+.1f}pt（{open_rate:.1f}%）と改善")

    if prev_click is not None and not pd.isna(prev_click):
        d = click_rate - prev_click
        if d <= -3:
            declines.append(f"クリック率が前週比 {d:+.1f}pt（{click_rate:.1f}%）と大幅に低下")
        elif d <= -1:
            declines.append(f"クリック率が前週比 {d:+.1f}pt（{click_rate:.1f}%）とやや低下")
        elif d >= 3:
            improvements.append(f"クリック率が前週比 {d:+.1f}pt（{click_rate:.1f}%）と大幅に改善")
        elif d >= 1:
            improvements.append(f"クリック率が前週比 {d:+.1f}pt（{click_rate:.1f}%）と改善")

    if prev_revenue is not None and not pd.isna(prev_revenue) and prev_revenue > 0:
        rev_change = (revenue - prev_revenue) / prev_revenue * 100
        if rev_change <= -30:
            declines.append(f"メール経由受注金額が ¥{revenue:,}（前週比 {rev_change:+.0f}%）と大幅減")
        elif rev_change <= -10:
            declines.append(f"メール経由受注金額が ¥{revenue:,}（前週比 {rev_change:+.0f}%）と減少")
        elif rev_change >= 30:
            improvements.append(f"メール経由受注金額が ¥{revenue:,}（前週比 {rev_change:+.0f}%）と大幅増")
        elif rev_change >= 10:
            improvements.append(f"メール経由受注金額が ¥{revenue:,}（前週比 {rev_change:+.0f}%）と増加")

    if declines:
        insights.append("【注意】" + "。".join(declines) + "。")
    if improvements:
        insights.append("【改善】" + "。".join(improvements) + "。")

    if not declines and not improvements:
        insights.append(f"全体: 配信成功 {delivered:,}人、開封率 {open_rate:.1f}%、クリック率 {click_rate:.1f}%、受注CVR {cvr:.1f}%（受注金額 ¥{revenue:,}）。")

    # 平均比のコメント
    avg_open = avg_d.get("開封率")
    if avg_open is not None and not pd.isna(avg_open):
        d = open_rate - avg_open
        if abs(d) >= 3:
            direction = "上回って" if d > 0 else "下回って"
            insights.append(f"開封率は過去4週平均（{avg_open:.1f}%）を {abs(d):.1f}pt {direction}いる。")

    # --- 2. 施策タイプ別インサイト ---
    if len(by_type) > 0:
        type_notes = []
        for _, tr in by_type.iterrows():
            ctype = tr.get("施策タイプ", "")
            cvr_def = CVR_DEFINITIONS.get(ctype, CVR_DEFINITIONS["EC購買"])
            target_cvr = tr.get("目標CVR", 0)
            delivered_t = int(tr.get("配信成功_人数", 0))

            if ctype == "EC購買" and delivered_t > 0:
                rev_t = int(tr.get("受注金額", 0))
                if target_cvr > 0:
                    type_notes.append(f"EC購買: 受注CVR {target_cvr:.1f}%、受注金額 ¥{rev_t:,}")
                elif delivered_t > 100:
                    type_notes.append(f"EC購買: 配信 {delivered_t:,}人だが受注CVR 0% — クリエイティブやオファーの見直しを検討")
            elif ctype == "CRM" and delivered_t > 0:
                if target_cvr >= 50:
                    type_notes.append(f"CRM: 開封CVR {target_cvr:.1f}% と高水準で関係維持に貢献")
                elif target_cvr >= 30:
                    type_notes.append(f"CRM: 開封CVR {target_cvr:.1f}%")
                elif delivered_t > 100:
                    type_notes.append(f"CRM: 開封CVR {target_cvr:.1f}% — 件名やタイミングの改善余地あり")
            elif ctype == "公式サイト流入" and delivered_t > 0:
                if target_cvr >= 3:
                    type_notes.append(f"公式サイト流入: クリックCVR {target_cvr:.1f}% と良好なサイト誘導")
                elif target_cvr >= 1:
                    type_notes.append(f"公式サイト流入: クリックCVR {target_cvr:.1f}%")
                elif delivered_t > 100:
                    type_notes.append(f"公式サイト流入: クリックCVR {target_cvr:.1f}% — CTA配置やコンテンツの改善を検討")

        if type_notes:
            insights.append("【タイプ別】" + "。".join(type_notes) + "。")

    # --- 3. 施策別ハイライト ---
    if len(by_campaign) > 0:
        bc = by_campaign.copy()
        if "開封率" in bc.columns and len(bc) > 1:
            best_open = bc.loc[bc["開封率"].idxmax()]
            name = str(best_open.get("施策名", ""))
            short_name = name if len(name) <= 30 else name[:30] + "…"
            insights.append(f"最高開封率:「{short_name}」{best_open['開封率']:.1f}%。")

        cart_rows = bc[bc["施策名"].str.contains("かご落ち", na=False)]
        if len(cart_rows) > 0:
            cart = cart_rows.iloc[0]
            cart_cvr = cart.get("受注CVR", 0)
            if cart_cvr > 0:
                insights.append(f"かご落ちメール: 受注CVR {cart_cvr:.1f}%（¥{int(cart.get('受注金額', 0)):,}）— 少数配信ながら高い転換率。")

    # --- 4. 年代別インサイト ---
    target_ages = ["20代", "30代", "40代"]
    if len(by_age) > 0 and "年代" in by_age.columns:
        target_data = by_age[by_age["年代"].isin(target_ages)]
        other_data = by_age[~by_age["年代"].isin(target_ages) & (by_age["年代"] != "不明")]
        if len(target_data) > 0 and len(other_data) > 0:
            t_open = target_data["開封率"].mean()
            o_open = other_data["開封率"].mean()
            gap = t_open - o_open
            if gap < -3:
                insights.append(f"ターゲット層（20〜40代）の開封率 {t_open:.1f}% はその他 {o_open:.1f}% より {abs(gap):.1f}pt 低い。件名やコンテンツのターゲット適合性を要検証。")
            elif gap > 3:
                insights.append(f"ターゲット層（20〜40代）の開封率 {t_open:.1f}% はその他 {o_open:.1f}% を {gap:.1f}pt 上回っており良好。")

    # --- 5. 許諾率インサイト ---
    if len(optin_health) > 0:
        total_optin = optin_health["オプトイン率"].mean()
        target_optin = optin_health[optin_health["年代"].isin(target_ages)]
        if len(target_optin) > 0:
            t_optin = target_optin["オプトイン率"].mean()
            if t_optin < 35:
                insights.append(f"ターゲット層の許諾率が平均 {t_optin:.1f}% と低水準。リスト拡大施策（会員登録時の許諾促進等）が課題。")

    return insights


def generate_html_report(summary, by_campaign, by_type, by_age, subject_rank,
                         hourly, optin_health, optin_trend_df,
                         week_start, week_end, prev, avg, yoy):

    s = summary.iloc[0] if len(summary) > 0 else {}
    prev_d = prev if prev else {}
    avg_d = avg if avg else {}
    yoy_d = yoy if yoy else {}

    # --- 要点生成 ---
    insights = generate_key_insights(summary, by_type, by_campaign, by_age,
                                      optin_health, subject_rank, prev, avg)

    # --- 比較データ有無の通知 ---
    parts = []
    if prev:
        parts.append("前週比")
    if avg:
        parts.append(f"過去{avg.get('_weeks', 0)}週平均比")
    if yoy:
        parts.append("前年同週比")
    if parts:
        comparison_note = f'<div class="note-info">{" / ".join(parts)} を表示中</div>'
    else:
        comparison_note = '<div class="note-info">初回実行のため比較データなし。次週以降、前週比・過去4週平均との比較が自動表示されます。</div>'

    # --- KPIカード ---
    kpis = [
        ("配信成功",        s.get("配信成功_人数", 0), "人（ユニーク）",  prev_d.get("配信成功_人数"), avg_d.get("配信成功_人数"), yoy_d.get("配信成功_人数"), False, True,  False),
        ("開封率",          s.get("開封率", 0),        "開封者 / 配信成功", prev_d.get("開封率"),        avg_d.get("開封率"),        yoy_d.get("開封率"),        True,  True,  True),
        ("クリック率",      s.get("クリック率", 0),    "クリック / 開封者", prev_d.get("クリック率"),    avg_d.get("クリック率"),    yoy_d.get("クリック率"),    True,  True,  True),
        ("受注CVR (3日)",   s.get("受注CVR", 0),       "受注 / クリック",  prev_d.get("受注CVR"),       avg_d.get("受注CVR"),       yoy_d.get("受注CVR"),       False, True,  True),
        ("開封CVR",         s.get("開封CVR", 0),       "開封 / 配信成功",  None,                        None,                       None,                       False, True,  True),
        ("クリックCVR",     s.get("クリックCVR", 0),   "クリック / 開封",  None,                        None,                       None,                       False, True,  True),
        ("メール経由受注金額", s.get("受注金額", 0),    "（税込・3日以内）", prev_d.get("受注金額"),     avg_d.get("受注金額"),      yoy_d.get("受注金額"),      False, True,  False),
        ("バウンス率",      s.get("バウンス率", 0),    "バウンス / 試行", prev_d.get("バウンス率"),    avg_d.get("バウンス率"),    yoy_d.get("バウンス率"),    False, False, True),
    ]
    kpi_html = '<div class="kpi-grid">\n'
    for label, val, unit, pv, av, yv, accent, hib, is_p in kpis:
        kpi_html += kpi_card_html(label, val, unit, pv, av, yv, accent, hib, is_p)
    kpi_html += '</div>'

    # --- 受注CVR ウィンドウ比較テーブル ---
    cvr_window_html = ""
    if len(summary) > 0:
        cvr_window_html = """
<h3>受注CVRアトリビューションウィンドウ比較</h3>
<table>
<thead><tr><th>ウィンドウ</th><th>受注_人数</th><th>受注金額</th><th>受注CVR</th></tr></thead>
<tbody>"""
        for label, suf in [("同日", "_同日"), ("3日以内", "_3日"), ("7日以内", "_7日")]:
            n = int(s.get(f"受注_人数{suf}", 0))
            a = int(s.get(f"受注金額{suf}", 0))
            cvr = s.get(f"受注CVR{suf}", 0)
            bold = ' style="font-weight:bold;background:#f0fdf4;"' if suf == "_3日" else ""
            star = " ★" if suf == "_3日" else ""
            cvr_window_html += f'<tr{bold}><td>{label}{star}</td><td>{n:,}</td><td>&yen;{a:,}</td><td class="highlight">{cvr:.1f}%</td></tr>\n'
        cvr_window_html += "</tbody></table>"

    # --- 施策タイプ別サマリー ---
    type_html = ""
    if len(by_type) > 0:
        type_html = """
<table>
<thead><tr><th>施策タイプ</th><th>目標CTA</th><th>配信成功</th><th>開封</th><th>開封率</th><th>クリック</th><th>クリック率</th><th>目標CVR</th><th>CVR種別</th></tr></thead>
<tbody>"""
        for _, r in by_type.iterrows():
            ctype = r.get("施策タイプ", "")
            cvr_def = CVR_DEFINITIONS.get(ctype, CVR_DEFINITIONS["EC購買"])
            cta_map = {"EC購買": "EC受注", "CRM": "開封（関係維持）", "公式サイト流入": "クリック（サイト誘導）"}
            type_html += f'<tr><td>{_type_badge(ctype)} {ctype}</td>'
            type_html += f'<td>{cta_map.get(ctype, "")}</td>'
            type_html += f'<td>{int(r.get("配信成功_人数", 0)):,}</td>'
            type_html += f'<td>{int(r.get("開封_人数", 0)):,}</td>'
            type_html += f'<td class="highlight">{r.get("開封率", 0):.1f}%</td>'
            type_html += f'<td>{int(r.get("クリック_人数", 0)):,}</td>'
            type_html += f'<td class="highlight">{r.get("クリック率", 0):.1f}%</td>'
            type_html += f'<td class="highlight">{r.get("目標CVR", 0):.1f}%</td>'
            type_html += f'<td>{cvr_def["label"]}</td>'
            type_html += '</tr>\n'
        type_html += "</tbody></table>"

    # --- 施策別テーブル ---
    camp_html = ""
    if len(by_campaign) > 0:
        camp_html = """
<table>
<thead>
<tr><th>施策タイプ</th><th>施策名</th><th>配信成功</th><th>開封</th><th>開封率</th><th>クリック</th><th>クリック率</th>
<th colspan="3">受注CVR</th><th>受注金額(3日)</th><th>目標CVR</th></tr>
<tr><th></th><th></th><th></th><th></th><th></th><th></th><th></th>
<th class="sub">同日</th><th class="sub">3日以内</th><th class="sub">7日以内</th><th></th><th></th></tr>
</thead>
<tbody>"""
        bc = by_campaign.copy()
        if "施策名" in bc.columns:
            bc["_type"] = bc["施策名"].apply(classify_campaign)
            bc["_type_sort"] = bc["_type"].apply(lambda x: CAMPAIGN_TYPE_ORDER.index(x) if x in CAMPAIGN_TYPE_ORDER else 99)
            bc = bc.sort_values(["_type_sort", "施策名"]).reset_index(drop=True)

        for _, r in bc.iterrows():
            ctype = classify_campaign(r.get("施策名", ""))
            cvr_def = CVR_DEFINITIONS.get(ctype, CVR_DEFINITIONS["EC購買"])
            if cvr_def["num"] == "受注":
                target_cvr = r.get("受注CVR", 0)
            elif cvr_def["num"] == "開封":
                target_cvr = r.get("開封CVR", 0)
            else:
                target_cvr = r.get("クリックCVR", 0)

            camp_html += f'<tr><td>{_type_badge(ctype)}</td>'
            camp_html += f'<td>{r.get("施策名", "")}</td>'
            camp_html += f'<td>{int(r.get("配信成功_人数", 0)):,}</td>'
            camp_html += f'<td>{int(r.get("開封_人数", 0)):,}</td>'
            camp_html += f'<td class="highlight">{r.get("開封率", 0):.1f}%</td>'
            camp_html += f'<td>{int(r.get("クリック_人数", 0)):,}</td>'
            camp_html += f'<td class="highlight">{r.get("クリック率", 0):.1f}%</td>'
            camp_html += f'<td>{r.get("受注CVR_同日", 0):.1f}%</td>'
            camp_html += f'<td class="highlight">{r.get("受注CVR_3日", 0):.1f}%</td>'
            camp_html += f'<td>{r.get("受注CVR_7日", 0):.1f}%</td>'
            camp_html += f'<td>&yen;{int(r.get("受注金額", 0)):,}</td>'
            camp_html += f'<td class="highlight">{target_cvr:.1f}% ({cvr_def["label"]})</td>'
            camp_html += '</tr>\n'
        camp_html += "</tbody></table>"

    # --- 年代別テーブル ---
    target_ages = ["20代", "30代", "40代"]
    age_rows = ""
    target_note = ""
    ad = pd.DataFrame()

    if len(by_age) > 0 and "年代" in by_age.columns:
        age_cols = ["年代", "配信成功_人数", "開封_人数", "開封率",
                    "クリック_人数", "クリック率", "受注_人数", "受注CVR"]
        ad = by_age[[c for c in age_cols if c in by_age.columns]].copy()
        ad["_sort"] = ad["年代"].apply(lambda x: AGE_ORDER.index(x) if x in AGE_ORDER else 99)
        ad = ad.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

        td_data = ad[ad["年代"].isin(target_ages)]
        nd_data = ad[~ad["年代"].isin(target_ages)]
        target_open = td_data["開封率"].mean() if len(td_data) > 0 else 0
        nontarget_open = nd_data["開封率"].mean() if len(nd_data) > 0 else 0

        for _, r in ad.iterrows():
            is_t = r["年代"] in target_ages
            bg = ' style="background:#fef2f2;"' if is_t else ""
            lbl = f'<strong>{r["年代"]}</strong>' if is_t else r["年代"]
            age_rows += f'<tr{bg}><td>{lbl}</td>'
            for col in ad.columns:
                if col == "年代":
                    continue
                val = r[col]
                if col in ["開封率", "クリック率", "受注CVR"]:
                    age_rows += f'<td class="highlight">{val:.1f}%</td>'
                elif isinstance(val, (int, float, np.integer, np.floating)):
                    age_rows += f'<td>{int(val):,}</td>'
                else:
                    age_rows += f'<td>{val}</td>'
            age_rows += "</tr>\n"

        target_note = f'<div class="note"><strong>ターゲット層（20〜40代）平均開封率: {target_open:.1f}%</strong> ／ その他: {nontarget_open:.1f}%{"　※ターゲット層の開封率が低い傾向" if target_open < nontarget_open else "　良好"}</div>'
    else:
        target_note = '<div class="note">会員データなしのため年代別分析はスキップ</div>'

    # --- 件名ランキング ---
    subj_rows = ""
    for i, (_, r) in enumerate(subject_rank.head(10).iterrows()):
        subj_rows += f'<tr><td>{i+1}</td><td>{r["メール件名"]}</td><td>{int(r["配信成功数"]):,}</td><td>{int(r["開封者数"]):,}</td><td class="highlight">{r["開封率"]:.1f}%</td></tr>\n'

    # --- 配信時間帯 ---
    hr_rows = ""
    for _, r in hourly.iterrows():
        hr_rows += f'<tr><td>{r["曜日"]}</td><td>{int(r["時間帯"])}:00</td><td>{int(r["配信成功数"]):,}</td><td>{int(r["開封者数"]):,}</td><td class="highlight">{r["開封率"]:.1f}%</td></tr>\n'

    # --- オプトイン ---
    optin_bars = ""
    max_rate = optin_health["オプトイン率"].max() if len(optin_health) > 0 else 100
    for _, r in optin_health.iterrows():
        w = int(r["オプトイン率"] / max(max_rate, 1) * 200)
        is_t = r["年代"] in target_ages
        color = "#e94560" if is_t else "#1a1a2e"
        b = "<strong>" if is_t else ""
        be = "</strong>" if is_t else ""
        optin_bars += f'<div class="bar-container"><div class="bar-label" style="width:70px">{b}{r["年代"]}{be}</div><div class="bar" style="width:{w}px;background:{color}"></div><div class="bar-label">{b}{r["オプトイン率"]:.1f}%{be} ({int(r["オプトイン数"]):,}/{int(r["会員数"]):,})</div></div>\n'

    optin_rows = ""
    for _, r in optin_health.iterrows():
        is_t = r["年代"] in target_ages
        bg = ' style="background:#fef2f2;"' if is_t else ""
        lbl = f'<strong>{r["年代"]}</strong>' if is_t else r["年代"]
        optin_rows += f'<tr{bg}><td>{lbl}</td><td>{int(r["会員数"]):,}</td><td>{int(r["オプトイン数"]):,}</td><td class="highlight">{r["オプトイン率"]:.1f}%</td></tr>\n'

    # --- 許諾率トレンドテーブル ---
    optin_trend_html = ""
    if optin_trend_df is not None and len(optin_trend_df) > 1:
        optin_trend_html = '<h3>過去4週 許諾率トレンド</h3>\n<table class="trend-table">\n<thead><tr><th>週</th><th>全体</th>'
        age_cols_trend = ["20代", "30代", "40代", "50代", "60代", "70代以上"]
        for a in age_cols_trend:
            optin_trend_html += f'<th>{a}</th>'
        optin_trend_html += '</tr></thead>\n<tbody>'

        prev_row_data = None
        for _, tr in optin_trend_df.iterrows():
            ws_label = tr["week_start"][:10]
            optin_trend_html += f'<tr><td>{ws_label}</td>'
            total_val = tr.get("許諾率_全体", 0)
            if pd.isna(total_val):
                total_val = 0
            optin_trend_html += f'<td><strong>{total_val:.1f}%</strong></td>'
            for a in age_cols_trend:
                col = f"許諾率_{a}"
                val = tr.get(col, 0)
                if pd.isna(val):
                    val = 0
                # 前週との差分で色付け
                cls = ""
                if prev_row_data is not None:
                    pv = prev_row_data.get(col, 0)
                    if pd.isna(pv):
                        pv = 0
                    if val > pv:
                        cls = ' class="trend-up"'
                    elif val < pv:
                        cls = ' class="trend-down"'
                optin_trend_html += f'<td{cls}>{val:.1f}%</td>'
            optin_trend_html += '</tr>\n'
            prev_row_data = tr.to_dict()
        optin_trend_html += '</tbody></table>'

    # --- 要点HTML ---
    if insights:
        insights_items = "\n".join(f'<li>{ins}</li>' for ins in insights)
        insights_html = f"""<div class="section">
<h2>要点</h2>
<ul style="margin:0;padding-left:20px;line-height:2.0;font-size:13px;">
{insights_items}
</ul>
</div>"""
    else:
        insights_html = ""

    # --- 組み立て ---
    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>メールマーケティング週次レポート {week_start}</title>
{CSS}
</head>
<body>
<div class="container">

<h1>メールマーケティング 週次レポート</h1>
<div class="meta">対象期間: {week_start} 〜 {week_end}　|　生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
{comparison_note}

{insights_html}

<div class="section">
<h2>1. 全体サマリー</h2>
{kpi_html}
{cvr_window_html}
</div>

<div class="section">
<h2>2. 施策タイプ別パフォーマンス</h2>
<div class="note-info">
施策タイプ別CVR定義:<br>
　<span class="type-badge type-ec">EC購買</span> 受注CVR = 受注数 ÷ クリック数（3日以内）<br>
　<span class="type-badge type-crm">CRM</span> 開封CVR = 開封数 ÷ 配信成功数<br>
　<span class="type-badge type-site">公式サイト流入</span> クリックCVR = クリック数 ÷ 開封数
</div>
{type_html}
</div>

<div class="section">
<h2>3. 施策別ブレイクダウン</h2>
{camp_html}
</div>

<div class="section">
<h2>4. 年代別セグメント分析</h2>
{target_note}
<table>
<thead><tr><th>年代</th>{"".join(f'<th>{c}</th>' for c in ad.columns if c != '年代') if len(ad) > 0 else ''}</tr></thead>
<tbody>{age_rows}</tbody>
</table>
</div>

<div class="section">
<h2>5. 件名別 開封率ランキング</h2>
<h3>上位10件</h3>
<table>
<thead><tr><th>#</th><th>メール件名</th><th>配信成功数</th><th>開封者数</th><th>開封率</th></tr></thead>
<tbody>{subj_rows}</tbody>
</table>
</div>

<div class="section">
<h2>6. 配信時間帯別 開封率</h2>
<table>
<thead><tr><th>曜日</th><th>時間帯</th><th>配信成功数</th><th>開封者数</th><th>開封率</th></tr></thead>
<tbody>{hr_rows}</tbody>
</table>
</div>

<div class="section">
<h2>7. メール受信許諾率（年代別）</h2>
<h3>有効会員のうち、メール配信を許諾している割合</h3>
<div style="margin-bottom:16px;">{optin_bars}</div>
<table>
<thead><tr><th>年代</th><th>有効会員数</th><th>メール許諾数</th><th>メール受信許諾率</th></tr></thead>
<tbody>{optin_rows}</tbody>
</table>
{optin_trend_html}
</div>

<div class="section">
<h2>算出定義</h2>
<table>
<tr><th>指標</th><th>計算式</th></tr>
<tr><td>開封率</td><td>開封者数（ユニーク） ÷ 配信成功数（ユニーク）</td></tr>
<tr><td>クリック率</td><td>クリック者数（ユニーク） ÷ 開封者数（ユニーク）</td></tr>
<tr><td>受注CVR</td><td>受注者数 ÷ クリック者数（同一顧客ID＋行動日から3日以内の受注で紐付け）</td></tr>
<tr><td>開封CVR（CRM用）</td><td>開封者数 ÷ 配信成功数（CRMメールの目標指標）</td></tr>
<tr><td>クリックCVR（サイト流入用）</td><td>クリック者数 ÷ 開封者数（公式サイト流入メールの目標指標）</td></tr>
<tr><td>バウンス率</td><td>配信失敗者数（ユニーク） ÷ 配信試行者数（ユニーク）</td></tr>
<tr><td>配信停止率</td><td>配信停止数 ÷ 配信成功数</td></tr>
<tr><td>前週比</td><td>今週値 − 前週値（pt または実数）</td></tr>
<tr><td>平均比</td><td>今週値 − 過去4週平均値</td></tr>
<tr><td>前年同週比</td><td>今週値 − 前年同週値（±1週で探索）</td></tr>
</table>
</div>

</div>
</body>
</html>
"""
    return html


# ====================================================================
# 6. メイン処理
# ====================================================================
def main():
    parser = argparse.ArgumentParser(description="藤崎 メールマーケティング週次レポート生成 v3")
    parser.add_argument("--week-start", default=None,
                        help="対象週の月曜日 (YYYY-MM-DD)。省略時は先週の月曜日")
    parser.add_argument("--log", default=None, help="メール行動ログCSV")
    parser.add_argument("--members", default=None, help="会員マスタCSV")
    parser.add_argument("--orders", default=None, help="受注データCSV")
    parser.add_argument("--output", default=None, help="出力HTMLファイル名")
    parser.add_argument("--history", default=None, help="履歴CSV (デフォルト: history.csv)")
    parser.add_argument("--encoding", default="cp932", help="CSV文字コード")
    parser.add_argument("--no-save", action="store_true", help="履歴CSVに追記しない")

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    if args.week_start:
        week_start = pd.Timestamp(args.week_start)
    else:
        today = pd.Timestamp.now()
        week_start = today - timedelta(days=today.weekday() + 7)
    week_end = week_start + timedelta(days=6)

    ws_str = week_start.strftime("%Y-%m-%d")
    we_str = week_end.strftime("%Y-%m-%d")
    week_key = week_start.strftime("%Y%m%d")

    data_dir = base_dir / "data" / week_key
    shared_dir = base_dir / "data" / "shared"

    def resolve_path(arg_val, filename):
        if arg_val:
            return arg_val
        week_path = data_dir / filename
        if week_path.exists():
            return str(week_path)
        shared_path = shared_dir / filename
        if shared_path.exists():
            return str(shared_path)
        return str(week_path)

    log_path = resolve_path(args.log, "メール行動ログ.csv")
    members_path = resolve_path(args.members, "会員マスタ.csv")
    orders_path = resolve_path(args.orders, "受注データ.csv")

    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output if args.output else str(output_dir / f"report_{week_key}.html")

    history_path = args.history if args.history else str(base_dir / "history.csv")

    if not Path(log_path).exists():
        shared_csvs = sorted(shared_dir.glob("*.csv"))
        if shared_csvs:
            print(f"⚠ 週フォルダにログなし → data/shared/ の {len(shared_csvs)} ファイルを結合して使用")
            log_path = "__shared_combined__"
        else:
            print(f"❌ メール行動ログ が見つかりません: {log_path}")
            print(f"   data/{week_key}/ または data/shared/ に CSV を配置してください。")
            sys.exit(1)

    members_optional = not Path(members_path).exists()
    orders_optional = not Path(orders_path).exists()

    print(f"📂 データ参照:")
    if log_path == "__shared_combined__":
        print(f"   ログ:   data/shared/ (複数ファイル結合)")
    else:
        print(f"   ログ:   {log_path}")
    print(f"   会員:   {members_path}{' (なし → スキップ)' if members_optional else ''}")
    print(f"   受注:   {orders_path}{' (なし → スキップ)' if orders_optional else ''}")

    df_log, df_mem, df_ord = load_data(
        log_path, members_path, orders_path, args.encoding,
        shared_dir=shared_dir,
        members_optional=members_optional,
        orders_optional=orders_optional,
    )
    log_w, ord_w = filter_week(df_log, df_ord, week_start)

    if len(log_w) == 0:
        print("⚠ 対象週のデータがありません。日付を確認してください。")
        sys.exit(1)

    # 施策タイプ分布を表示
    type_counts = log_w.groupby("施策タイプ")["施策名"].nunique()
    print(f"  施策タイプ分布: {dict(type_counts)}")

    print("[3/6] 集計中...")
    summary = compute_funnel(log_w, ord_w)
    by_campaign = compute_funnel(log_w, ord_w, group_cols=["施策名"])
    by_type = compute_funnel_by_type(log_w, ord_w)
    if "年代" in log_w.columns and log_w["年代"].notna().any():
        by_age = compute_funnel(log_w, ord_w, group_cols=["年代"])
    else:
        by_age = pd.DataFrame()
    subject_rank = compute_subject_ranking(log_w)
    hourly = compute_hourly_heatmap(log_w)
    optin_health = compute_optin_health(df_mem)
    optin_total = compute_optin_total(df_mem)

    print("[4/6] 履歴データ読み込み中...")
    history_df = load_history(history_path)
    prev, avg, yoy = get_comparison(history_df, ws_str)
    if prev:
        print(f"  前週データあり → 前週比を表示")
    else:
        print(f"  前週データなし")
    if avg:
        print(f"  過去{avg.get('_weeks', 0)}週分の平均データあり")
    if yoy:
        print(f"  前年同週データあり → 前年同週比を表示")
    else:
        print(f"  前年同週データなし")

    # 履歴保存（HTML生成前に保存して、トレンドに今週分を含める）
    if not args.no_save:
        history_df = save_history(history_path, history_df, summary, ws_str,
                                  optin_health=optin_health, optin_total=optin_total)

    optin_trend_df = get_optin_trend(history_df, ws_str, n_weeks=4)

    print("[5/6] HTMLレポート生成中...")
    html = generate_html_report(
        summary, by_campaign, by_type, by_age, subject_rank,
        hourly, optin_health, optin_trend_df,
        ws_str, we_str, prev, avg, yoy
    )

    print(f"[6/6] 書き出し: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ 完了！ {output_path} をブラウザで開いてください。")

    if len(summary) > 0:
        s = summary.iloc[0]
        print(f"\n--- 週次サマリー ({ws_str} 〜 {we_str}) ---")
        print(f"  配信成功:   {int(s.get('配信成功_人数', 0)):,} 人")
        print(f"  開封率:     {s.get('開封率', 0):.1f}%", end="")
        if prev:
            print(f"  (前週比 {fmt_delta(s.get('開封率',0), prev.get('開封率'))})", end="")
        print()
        print(f"  クリック率: {s.get('クリック率', 0):.1f}%", end="")
        if prev:
            print(f"  (前週比 {fmt_delta(s.get('クリック率',0), prev.get('クリック率'))})", end="")
        print()
        print(f"  受注CVR(3日): {s.get('受注CVR', 0):.1f}%")
        print(f"  受注金額(3日): ¥{int(s.get('受注金額', 0)):,}")


if __name__ == "__main__":
    main()
