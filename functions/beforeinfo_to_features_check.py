import pandas as pd
import numpy as np
import datetime
import re
import download_pred
import wakamatsu_off_beforeinfo_html_to_csv
from datetime import datetime, timedelta, timezone

def create_race_key(df: pd.DataFrame) -> pd.Series:
    """
    stadium, race_date, race_noからユニークなrace_keyを生成する。
    SQLのcore.f_race_key関数に相当。
    """
    return (pd.to_datetime(df['race_date']).dt.strftime('%Y-%m-%d') + '-' +
            df['race_no'].astype(str) + '-' + df['stadium'].astype(str))

def create_core_pred_boat_info(beforeinfo_df: pd.DataFrame) -> pd.DataFrame:
    """
    SQLのcore.pred_boat_infoビューを再現。
    """
    df = beforeinfo_df.copy()
    df['race_key'] = create_race_key(df)

    # weight_rawから数値のみを抽出
    df['weight'] = pd.to_numeric(df['weight'].str.replace(r'[^0-9.]', '', regex=True), errors='coerce')

    # STからfs_flagとbf_st_timeを生成
    df['fs_flag'] = df['ST'].str.startswith('F', na=False)
    
    st_time_numeric = df['ST'].str.replace('F', '', regex=False)
    st_time_numeric = pd.to_numeric(st_time_numeric, errors='coerce')
    
    df['bf_st_time'] = np.where(df['fs_flag'], -st_time_numeric, st_time_numeric)

    # ▼▼▼ ここに修正を追加 ▼▼▼
    # exhibition_timeとtiltもこの段階で数値型に変換しておく
    df['exhibition_time'] = pd.to_numeric(df['exhibition_time'], errors='coerce')
    df['tilt'] = pd.to_numeric(df['tilt'], errors='coerce')
    # ▲▲▲ ここまで ▲▲▲

    # SQLの `DISTINCT ON (b.race_key, b.lane) ... ORDER BY b.course` を再現
    df_sorted = df.sort_values(['race_key', 'lane', 'course'])
    core_boat_info = df_sorted.drop_duplicates(subset=['race_key', 'lane'], keep='first')
    
    # カラム名をbf_courseに変更
    core_boat_info = core_boat_info.rename(columns={'course': 'bf_course'})

    return core_boat_info[[
        'race_key', 'lane', 'racer_id', 'weight', 'exhibition_time', 'tilt',
        'fs_flag', 'bf_st_time', 'bf_course'
    ]]

def create_core_pred_weather(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    SQLのcore.pred_weatherビューを再現。
    """
    df = weather_df.copy()
    df['race_key'] = create_race_key(df)

    # 各_rawカラムから数値のみを抽出
    cols_to_convert = ['air_temp_C', 'wind_speed_m', 'wave_height_cm', 'water_temp_C']
    for col in cols_to_convert:
        # 文字列から数字とドット以外を削除
        cleaned_series = df[col].str.replace(r'[^0-9.]', '', regex=True)
        # 数値型に変換（変換できない値はNaNになる）
        df[col] = pd.to_numeric(cleaned_series, errors='coerce')

    # wind_dir_rawからwind_dir_degへ変換
    wind_dir_map = {
        'is-wind1': 22.5,
        'is-wind2': 45,
        'is-wind3': 67.5,
        'is-wind4': 90,
        'is-wind5': 112.5,
        'is-wind6': 135,
        'is-wind7': 157.5,
        'is-wind8': 180,
        'is-wind9': 202.5,
        'is-wind10': 225,
        'is-wind11': 247.5,
        'is-wind12': 270,
        'is-wind13': 292.5,
        'is-wind14': 315,
        'is-wind15': 337.5,
        'is-wind16': 0
    }
    df['wind_dir_deg'] = df['wind_dir_icon'].map(wind_dir_map)

    # SQLの `DISTINCT ON (race_key) ... ORDER BY obs_time_label DESC` を再現
    df_sorted = df.sort_values(['race_key'], ascending=[True])
    core_weather = df_sorted.drop_duplicates(subset='race_key', keep='first')

    return core_weather[[
        'race_key', 'air_temp_C', 'wind_speed_m', 'wind_dir_deg',
        'wave_height_cm', 'water_temp_C', 'weather'
    ]]

def create_pred_features(boat_info_df: pd.DataFrame, weather_df: pd.DataFrame, beforeinfo_df: pd.DataFrame) -> pd.DataFrame:
    """
    SQLのpred.featuresビューを再現。
    """
    # 1. pred.boat_flatの作成
    boat_flat_df = pd.merge(boat_info_df, weather_df, on='race_key', how='left')

    # 2. core.pred_racesの作成
    races_df = beforeinfo_df[['race_date', 'stadium', 'race_no']].drop_duplicates()
    races_df['race_key'] = create_race_key(races_df)
    races_df = races_df[['race_key', 'race_date']]
    
    # 3. flatの作成
    flat_df = pd.merge(boat_flat_df, races_df, on='race_key')
    
    # 4. ピボット処理による横持ち化
    # まず、共通の天候・日付情報を集約
    base_features = flat_df.groupby('race_key').agg(
        race_date=('race_date', 'first'),
        air_temp_C=('air_temp_C', 'first'),
        wind_speed_m=('wind_speed_m', 'first'),
        wave_height_cm=('wave_height_cm', 'first'),
        water_temp_C=('water_temp_C', 'first'),
        weather_txt=('weather', 'first'),
        wind_dir_deg=('wind_dir_deg', 'first')
    ).reset_index()

    # レーンごとの情報をピボット
    pivot_df = flat_df.pivot_table(
        index='race_key',
        columns='lane',
        values=['racer_id', 'weight', 'exhibition_time', 'bf_st_time', 'bf_course', 'fs_flag'],
        aggfunc='first'
    )
    # マルチレベルカラムをフラット化
    pivot_df.columns = [f'lane{col[1]}_{col[0]}' for col in pivot_df.columns]
    pivot_df.reset_index(inplace=True)

    # 欠損しているレーン列をNaNで補完（常に1〜6が揃うようにする）
    required_fields = ['racer_id', 'weight', 'exhibition_time', 'bf_st_time', 'bf_course', 'fs_flag']
    for i in range(1, 7):
        for f in required_fields:
            col = f'lane{i}_{f}'
            if col not in pivot_df.columns:
                pivot_df[col] = np.nan

    # ベース情報とピボット情報を結合
    features_df = pd.merge(base_features, pivot_df, on='race_key')

    return features_df

def create_pred_tf2_lane_stats(features_df: pd.DataFrame, filtered_course_df: pd.DataFrame) -> pd.DataFrame:
    """
    SQLのpred.tf2_lane_statsビューを再現。
    """
    # 1. tf2_longの作成 (ワイドからロングへ)
    id_vars = ['race_key']
    value_vars_racer = [f'lane{i}_racer_id' for i in range(1, 7)]
    value_vars_course = [f'lane{i}_bf_course' for i in range(1, 7)]

    # features_dfに存在する列だけを対象にする（欠損列があってもエラーにしない）
    value_vars_racer = [c for c in value_vars_racer if c in features_df.columns]
    value_vars_course = [c for c in value_vars_course if c in features_df.columns]

    # どちらかが空なら統計は作れないので空データフレームを返す
    if not value_vars_racer or not value_vars_course:
        return pd.DataFrame({'race_key': features_df['race_key'].unique()})
    
    df_racer = pd.melt(features_df, id_vars=id_vars, value_vars=value_vars_racer, var_name='lane_no_str_racer', value_name='reg_no')
    df_course = pd.melt(features_df, id_vars=id_vars, value_vars=value_vars_course, var_name='lane_no_str_course', value_name='course')

    df_racer['lane_no'] = df_racer['lane_no_str_racer'].str.extract(r'lane(\d)_').astype(int)
    df_course['lane_no'] = df_course['lane_no_str_course'].str.extract(r'lane(\d)_').astype(int)
    
    tf2_long = pd.merge(df_racer[['race_key', 'lane_no', 'reg_no']], 
                        df_course[['race_key', 'lane_no', 'course']], 
                        on=['race_key', 'lane_no'])
    
    # 安全策：結合前にreg_no/courseの欠損を除去
    tf2_long = tf2_long.dropna(subset=['reg_no', 'course'])

    # --- ▼▼▼ ここから修正 ▼▼▼ ---
    # 結合キーに欠損値がある行を削除
    tf2_long.dropna(subset=['reg_no', 'course'], inplace=True)

    # 結合キーのデータ型を整数に変換して統一する
    tf2_long['reg_no'] = tf2_long['reg_no'].astype(int)
    tf2_long['course'] = tf2_long['course'].astype(int)
    # --- ▲▲▲ ここまで修正 ▲▲▲ ---

    # target_venues = features_df['jcd'].unique()

    # # 2. 取得した会場コードのリストを使い、filtered_course_df を絞り込む
    # #    .isin() を使うことで、「target_venuesリストに含まれるjcdを持つ行」だけを抽出できます。
    # filtered_df = filtered_course_df[filtered_course_df['jcd'].isin(target_venues)]


    # # --- 絞り込んだデータフレームを関数に渡す ---

    # # 以前の filtered_course_df の代わりに、絞り込んだ filtered_df を渡します。
    # pred_stats_df = create_pred_tf2_lane_stats(features_df, filtered_df)

    # 2. filtered_courseを結合
    merged_df = pd.merge(tf2_long, filtered_course_df, on=['reg_no', 'course'], how='left')

    # 3. 再度ピボットして横持ち化
    stats_pivot = merged_df.pivot_table(
        index='race_key',
        columns='lane_no',
        values=['starts', 'firsts', 'first_rate', 'two_rate', 'three_rate'],
        aggfunc='first'  # 念のため'first'を指定
    )
    stats_pivot.columns = [f'lane{col[1]}_{col[0]}' for col in stats_pivot.columns]
    stats_pivot.reset_index(inplace=True)
    
    # 欠けているlane統計列をNaNで補完（下流の結合で列構成を安定化）
    stat_fields = ['starts', 'firsts', 'first_rate', 'two_rate', 'three_rate']
    for i in range(1, 7):
        for f in stat_fields:
            col = f'lane{i}_{f}'
            if col not in stats_pivot.columns:
                stats_pivot[col] = np.nan
    
    return stats_pivot

def rename_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    prefix = {
        'air_temp_C': 'air_temp',
        'wind_speed_m': 'wind_speed',
        'wave_height_cm': 'wave_height',
        'water_temp_C': 'water_temp',
        'weather': 'weather_txt',
        'wind_dir': 'wind_dir_deg',
        **{f'lane{i}_exhibition_time': f'lane{i}_exh_time' for i in range(1, 7)},
    }
    df.rename(columns=prefix, inplace=True)
    df['air_temp'] = None
    df['water_temp'] = None
    df['wave_height'] = df['wave_height'] * 0.01 
    return df

# --- メイン処理 ---
def main(rno: int, jcd: str, path: str=None) -> pd.DataFrame:
    # 1. CSVファイルの読み込み
    try:
        today = datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d')
        if path:
            with open(path, 'r', encoding='utf-8') as file:
                html = file.read()
            beforeinfo_df, weather_df, meta_df = wakamatsu_off_beforeinfo_html_to_csv.parse_boat_race_html(html, is_pred=True)
        else:
            html = download_pred.download_off_pred(jcd, today, rno=rno)
            # html_path = 'wakamatsu_beforeinfo_20_20250830_8.html'  # ローカルのHTMLファイルパス
            # with open(html_path, 'r', encoding='utf-8') as file:
            #     html = file.read()
            beforeinfo_df, weather_df, meta_df = wakamatsu_off_beforeinfo_html_to_csv.parse_boat_race_html(html, is_pred=True)
        print(beforeinfo_df.head())
        print(weather_df.head())
        if beforeinfo_df.empty or weather_df.empty or meta_df.empty:
            print("⚠️ 必要なデータが取得できませんでした。処理を終了します。")
            return pd.DataFrame()  # 空のDataFrameを返す
        filtered_course_df = pd.read_csv('filtered_course.csv')
        studium = meta_df.iloc[0]['place']
        beforeinfo_df['race_no'] = rno
        beforeinfo_df['stadium'] = studium
        beforeinfo_df['race_date'] = today
        weather_df['stadium'] = studium
        weather_df['race_date'] = today
        weather_df['race_no'] = rno
        filtered_course_df = filtered_course_df[filtered_course_df['stadium'] == studium]
        # filtered_course_df.to_csv('filtered_course_若松.csv', index=False)
        print(filtered_course_df.head())
    except FileNotFoundError as e:
        print(f"エラー: {e}")
        print("必要なCSVファイル（beforeinfo.csv, weather.csv, filtered_course.csv）をコードと同じディレクトリに配置してください。")
        exit()

    # 2. core層ビューの作成
    core_boat_info = create_core_pred_boat_info(beforeinfo_df)
    core_weather = create_core_pred_weather(weather_df)

    # 3. features層ビューの作成
    features = create_pred_features(core_boat_info, core_weather, beforeinfo_df)
    
    # 4. tf2_lane_stats層ビューの作成
    tf2_lane_stats = create_pred_tf2_lane_stats(features, filtered_course_df)

    # 5. 最終的なfeatures_with_recordの作成
    features_with_record = pd.merge(features, tf2_lane_stats, on='race_key', how='left')

    # 6. カラム名のリネーム
    features_with_record = rename_columns(features_with_record, prefix={})
    # 結果の表示
    print("--- 最終成果物: features_with_record (先頭5行) ---")
    print(features_with_record.head())

    # 結果をCSVとして保存
    return features_with_record

if __name__ == "__main__":
    main()