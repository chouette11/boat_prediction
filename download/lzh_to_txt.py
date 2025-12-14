import os
import lhafile
import pathlib

def lzh_to_txt(lzh_path: str, base_out_dir: str, rel_path: str):
    lz = lhafile.Lhafile(lzh_path)
    full_out_dir = os.path.join(base_out_dir, rel_path)
    for info in lz.infolist():
        name = info.filename  # e.g., k250815.txt / b250815.txt
        raw = lz.read(name)  # バイト列（SJIS想定）
        # まずは Windows-31J/CP932 でデコード。失敗時は Shift_JIS にフォールバック
        try:
            text = raw.decode("cp932")
        except UnicodeDecodeError:
            text = raw.decode("shift_jis", errors="replace")

        os.makedirs(full_out_dir, exist_ok=True)
        out_path = os.path.join(full_out_dir, name)
        # UTF-8 でテキストとして保存（改行はそのまま）
        with open(out_path, "w", encoding="utf-8", newline="") as f_out:
            f_out.write(text)

# 解凍例：ある月をまとめて解凍
if __name__ == "__main__":
    import datetime as dt
    last_month_date = dt.date.today() - dt.timedelta(days=30)
    results_root = f"download/lzh/results/{last_month_date.year}/11"
    programs_root = f"download/lzh/programs/{last_month_date.year}/11"
    pathlib.Path("download/txt/results").mkdir(parents=True, exist_ok=True)
    pathlib.Path("download/txt/programs").mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(results_root):
        for file in files:
            if file.endswith(".lzh"):
                rel_path = os.path.relpath(root, results_root)
                lzh_to_txt(os.path.join(root, file), "download/txt/results", rel_path)
    for root, dirs, files in os.walk(programs_root):
        for file in files:
            if file.endswith(".lzh"):
                rel_path = os.path.relpath(root, programs_root)
                lzh_to_txt(os.path.join(root, file), "download/txt/programs", rel_path)
    print("解凍完了")
    print("results と programs の lzh をそれぞれ download/txt/results, download/txt/programs に解凍しました。")
    print("lzh_to_txt.py を実行して、lzh ファイルをテキストファイルに変換しました。")
