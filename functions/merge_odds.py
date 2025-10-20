import pandas as pd

def merge_odds(eval_df, result_df):
    """
    Attach 'trifecta_odds' and 'trio_odds' from eval_df to result_df on 'race_key'.
    eval_df must contain columns: 'race_key', 'trifecta_odds', 'trio_odds'.
    """
    # Prepare source odds
    odds_df = eval_df[["race_key", "trifecta_odds", "trio_odds"]].copy()

    # Ensure join key types match to avoid implicit dtype mismatches
    if "race_key" in result_df.columns:
        result_df = result_df.copy()
        result_df["race_key"] = result_df["race_key"].astype(str)
    odds_df["race_key"] = odds_df["race_key"].astype(str)

    # Avoid merge suffix collisions when result_df already has odds columns from a prior run
    drop_cols = [
        "trifecta_odds", "trio_odds",
        "trifecta_odds_x", "trifecta_odds_y",
        "trio_odds_x", "trio_odds_y",
    ]
    existing = [c for c in drop_cols if c in result_df.columns]
    if existing:
        result_df = result_df.drop(columns=existing)

    # If eval_df has duplicated race_key, keep the last occurrence (spec is OK to drop)
    odds_df = odds_df.drop_duplicates(subset=["race_key"], keep="last")

    # Left-join odds into results
    merged_df = pd.merge(result_df, odds_df, on="race_key", how="left")
    return merged_df

if __name__ == "__main__":
    jcd_name_dict = {
        "01": "桐 生",
        "07": "蒲 郡",
        "12": "住之江",
        "15": "丸 亀",
        "19": "下 関",
        "20": "若 松",
        "24": "大 村",
    }
    jcd = "07"
    csv_path = f"../model/artifacts/{jcd_name_dict.get(jcd)}/eval_features_recent_{jcd_name_dict.get(jcd)}.csv"
    result_path = "07_predictions_20251016_171136.csv"
    eval_df = pd.read_csv(csv_path)
    result_df = pd.read_csv(result_path)
    merged_df = merge_odds(eval_df, result_df)
    print(merged_df.head())
    # Set odds to 0 where trifecta_is_hit and trio_is_hit are not True
    merged_df["trifecta_odds"] = merged_df.apply(
        lambda row: row["trifecta_odds"] if bool(row.get("trifecta_is_hit", False)) else 0,
        axis=1
    )
    merged_df["trio_odds"] = merged_df.apply(
        lambda row: row["trio_odds"] if bool(row.get("trio_is_hit", False)) else 0,
        axis=1
    )
    print(merged_df.head())
    merged_df.to_csv(result_path, index=False)
