# ===== Result fetching & daily summary helpers =====
import os
import re
import requests
from datetime import timedelta, timezone
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup

JST = timezone(timedelta(hours=9))
STAKE_YEN_PER_SIGNAL = int(os.getenv("STAKE_YEN_PER_SIGNAL", "500"))  # 1点100円を既定

def tidy(text: str | None) -> str:
    "Collapse whitespace and strip."
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text).strip()

def parse_payouts_table(soup: BeautifulSoup) -> list[dict[str, str]]:
    "Extract bet‑type payout rows."
    payouts: list[dict[str, str]] = []
    # Table whose header begins with '勝式'
    table = None
    for t in soup.select("table.is-w495"):
        th = tidy(t.select_one("thead th").get_text()) if t.select_one("thead th") else ""
        if th == "勝式":
            table = t
            break
    if table is None:
        return payouts

    for tr in table.select("tbody tr"):
        cells = tr.find_all("td")
        if len(cells) < 4:
            continue
        bet_type = tidy(cells[0].get_text())
        # Skip rows where bet_type cell is empty (continuation rows)
        if not bet_type:
            continue
        comb = tidy(cells[1].get_text().replace("\u00a0", " "))
        payout = tidy(cells[2].get_text().replace("¥", "").replace("￥", "").replace(",", ""))
        popularity = tidy(cells[3].get_text())
        payouts.append(
            {
                "bet_type": bet_type,
                "combination": comb,
                "payout_yen": payout,
                "popularity": popularity,
            }
        )
    return payouts

def _fetch_official_result(url: str, timeout_sec: int = 12) -> tuple[str | None, int | None]:
    print("ダウンロード中:", url)
    """
    公式結果ページをダウンロードして三連複の結果と払い戻しを返す。
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=timeout_sec)
        if resp.status_code != 200:
            print(f"[result] GET {url} -> {resp.status_code}")
            return None, None
        soup = BeautifulSoup(resp.text, "html.parser")
        # まずは payouts table を探す
        payouts = parse_payouts_table(soup)
        print("抽出された払戻データ:", payouts)
        tri = payouts[0].get("combination")
        payout_yen = int(payouts[1].get("payout_yen", "0")) if payouts[1].get("payout_yen") else None
        return tri, payout_yen
    except Exception as e:
        print(f"[result] fetch failed for {url}: {e}")
        return None, None

def _evaluate_and_update_race(doc_id: str, data: dict, DB) -> dict:
    """
    race_notifications/{doc_id} の通知済みレース1件について、公式結果を評価し
    Firestoreに 'official_tri', 'payout_yen', 'hit', 'pl_yen' 等を保存して返す。
    既に official_tri/payout_yen が保存済みならダウンロードを省略する。
    戻り値は評価済みフィールドの dict（未評価・未確定の場合は空 dict）。
    """
    result_url = data.get("result_url")
    pred_tri = data.get("tri")
    hd = data.get("hd")
    print(f"評価中: {doc_id}, 予想三連単: {pred_tri}, 結果URL: {result_url}")
    if not result_url or not pred_tri or not hd:
        return {}

    # 既に評価済みならそのまま返す
    if all(k in data for k in ("official_tri", "payout_yen", "hit", "pl_yen")):
        return {
            "official_tri": data.get("official_tri"),
            "payout_yen": data.get("payout_yen"),
            "hit": data.get("hit"),
            "pl_yen": data.get("pl_yen"),
        }

    official_tri, payout = _fetch_official_result(result_url)
    if official_tri is None:
        # まだ結果ページ未確定など
        print(f"[result] Not ready for {doc_id}")
        return {}

    # hit条件を三連複に変更
    hit = set(pred_tri.split("-")) == set(official_tri.split("-"))
    stake = STAKE_YEN_PER_SIGNAL
    ret = int(payout or 0) if hit else 0
    pl = ret - stake

    payload = {
        "official_tri": official_tri,
        "payout_yen": int(payout or 0),
        "stake_yen": int(stake),
        "hit": bool(hit),
        "pl_yen": int(pl),
    }
    try:
        DB.collection("race_notifications").document(doc_id).set(payload, merge=True)
    except Exception as e:
        print(f"[firestore] failed to update result fields for {doc_id}: {e}")
    return payload

def _build_daily_summary(hd: str, DB) -> dict:
    print("DBから日次サマリ構築中:", DB)
    print("プロジェクト", DB._database_string)  # type: ignore
    """
    指定日の通知済みレース（sent==True）について、未評価分は評価してから日次サマリを構築。
    戻り値: {count_total, count_completed, hits, staked_yen, returned_yen, pnl_yen, hit_rate, roi}
    """
    try:
        q = DB.collection("race_notifications").where("hd", "==", hd).where("sent", "==", True)
        snaps = list(q.stream())
    except Exception as e:
        print(f"[firestore] query failed in _build_daily_summary: {e}")
        snaps = []

    total = len(snaps)
    hits = 0
    staked = 0
    returned = 0
    completed = 0

    for snap in snaps:
        data = snap.to_dict() or {}
        doc_id = snap.id

        # 評価を実行（未評価で結果が出ていれば official/payout を取得できる）
        eval_fields = _evaluate_and_update_race(doc_id, data, DB)
        if not eval_fields:
            # まだ結果が出ていない
            continue

        completed += 1
        staked += STAKE_YEN_PER_SIGNAL
        if eval_fields.get("hit"):
            hits += 1
            returned += int(eval_fields.get("payout_yen", 0)) * (STAKE_YEN_PER_SIGNAL // 100)

    pnl = returned - staked
    hit_rate = (hits / completed) if completed else 0.0
    roi = (returned / staked) if staked else 0.0

    return {
        "count_total": int(total),
        "count_completed": int(completed),
        "hits": int(hits),
        "staked_yen": int(staked),
        "returned_yen": int(returned),
        "pnl_yen": int(pnl),
        "hit_rate": float(hit_rate),
        "roi": float(roi),
    }

def _format_summary_message(hd: str, summary: dict) -> str:
    y = hd[:4]; m = hd[4:6]; d = hd[6:8]
    total = summary.get("count_total", 0)
    done = summary.get("count_completed", 0)
    hits = summary.get("hits", 0)
    staked = summary.get("staked_yen", 0)
    returned = summary.get("returned_yen", 0)
    pnl = summary.get("pnl_yen", 0)
    hit_rate = summary.get("hit_rate", 0.0) * 100.0
    roi = summary.get("roi", 0.0)

    status = "（未確定あり）" if done < total else ""
    lines = [
        f"【日次サマリ {y}/{m}/{d}{status}】",
        f"通知レース: {total}件 / 確定: {done}件 / 的中: {hits}件",
        f"投資: {staked:,}円 / 回収: {returned:,}円 / 収支: {pnl:+,}円",
        f"的中率: {hit_rate:.1f}% / ROI: {roi:.3f}",
    ]
    return "\n".join(lines)

def _aggregate_daily_summaries(start_date, end_date, DB):
    """
    daily_summaries/{hd} を start_date..end_date (両端含む) で合算する。
    戻り値は日次と同じキー構成。
    """
    total = completed = hits = staked = returned = 0
    cur = start_date
    while cur <= end_date:
        hd = cur.strftime("%Y%m%d")
        try:
            snap = DB.collection("daily_summaries").document(hd).get()
            if snap.exists:
                s = snap.to_dict() or {}
                total += int(s.get("count_total", 0))
                completed += int(s.get("count_completed", 0))
                hits += int(s.get("hits", 0))
                staked += int(s.get("staked_yen", 0))
                returned += int(s.get("returned_yen", 0))
        except Exception as e:
            print(f"[firestore] read daily summary for {hd} failed: {e}")
        cur += timedelta(days=1)

    pnl = returned - staked
    hit_rate = (hits / completed) if completed else 0.0
    roi = (returned / staked) if staked else 0.0
    return {
        "count_total": int(total),
        "count_completed": int(completed),
        "hits": int(hits),
        "staked_yen": int(staked),
        "returned_yen": int(returned),
        "pnl_yen": int(pnl),
        "hit_rate": float(hit_rate),
        "roi": float(roi),
    }

def _format_period_summary_message(title: str, start_date, end_date, summary: dict) -> str:
    lines = [
        f"【{title} {start_date.strftime('%Y/%m/%d')}–{end_date.strftime('%Y/%m/%d')}】",
        f"通知レース: {summary.get('count_total', 0)}件 / 確定: {summary.get('count_completed', 0)}件 / 的中: {summary.get('hits', 0)}件",
        f"投資: {summary.get('staked_yen', 0):,}円 / 回収: {summary.get('returned_yen', 0):,}円 / 収支: {summary.get('pnl_yen', 0):+,}円",
        f"的中率: {summary.get('hit_rate', 0.0)*100:.1f}% / ROI: {summary.get('roi', 0.0):.3f}",
    ]
    return "\n".join(lines)

def daily_main(hd: str, DB) -> str | None:
    summary = _build_daily_summary(hd, DB)
    message = _format_summary_message(hd, summary)
    return summary, message

def weekly_main(week_start_date, week_end_date, DB) -> str | None:
    summary = _aggregate_daily_summaries(week_start_date, week_end_date, DB)
    message = _format_period_summary_message("週間サマリ", week_start_date, week_end_date, summary)
    return summary, message

def monthly_main(month_start_date, month_end_date, DB) -> str | None:
    summary = _aggregate_daily_summaries(month_start_date, month_end_date, DB)
    message = _format_period_summary_message("月間サマリ", month_start_date, month_end_date, summary)
    return summary, message