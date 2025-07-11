"""
Preprocess module for Boat Race CPL-Net features.
Train mode:   fit scaler / save artifacts / output parquet
Inference mode: load scaler / transform / output parquet
"""

import argparse, os, joblib, pandas as pd
from sklearn.preprocessing import StandardScaler

NUMERIC_COLS = ["wind_speed", "wave_height"]

def fit_transform_train(df, out_dir="artifacts"):
    os.makedirs(out_dir, exist_ok=True)
    scaler = StandardScaler()
    df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])
    joblib.dump(scaler, f"{out_dir}/scaler.pkl")
    df.fillna(-1, inplace=True)
    return df

def transform_infer(df, in_dir="artifacts"):
    scaler = joblib.load(f"{in_dir}/scaler.pkl")
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
    df.fillna(-1, inplace=True)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--mode", choices=["train", "infer"], default="train")
    ap.add_argument("--out",  default="processed.parquet")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.mode == "train":
        df = fit_transform_train(df)
    else:
        df = transform_infer(df)
    df.to_parquet(args.out, index=False)
    print(f"✔ saved {args.out}")

if __name__ == "__main__":
    main()
