import pandas as pd
import numpy as np
import datetime
import re
import download_pred
import wakamatsu_off_beforeinfo_html_to_csv

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
    for col in ['air_temp_C', 'wind_speed_m', 'wave_height_cm', 'water_temp_C']:
        df[col] = float(df[col].str.replace(r'[^0-9.]', '', regex=True))

    # wind_dir_rawからwind_dir_degへ変換
    wind_dir_map = {
        'is-wind1': 0,
        'is-wind2': 22.5,
        'is-wind3': 45,
        'is-wind4': 67.5,
        'is-wind5': 90,
        'is-wind6': 112.5,
        'is-wind7': 135,
        'is-wind8': 157.5,
        'is-wind9': 180,
        'is-wind10': 202.5,
        'is-wind11': 225,
        'is-wind12': 247.5,
        'is-wind13': 270,
        'is-wind14': 292.5,
        'is-wind15': 315,
        'is-wind16': 337.5
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
    
    df_racer = pd.melt(features_df, id_vars=id_vars, value_vars=value_vars_racer, var_name='lane_no_str_racer', value_name='reg_no')
    df_course = pd.melt(features_df, id_vars=id_vars, value_vars=value_vars_course, var_name='lane_no_str_course', value_name='course')

    df_racer['lane_no'] = df_racer['lane_no_str_racer'].str.extract(r'lane(\d)_').astype(int)
    df_course['lane_no'] = df_course['lane_no_str_course'].str.extract(r'lane(\d)_').astype(int)
    
    tf2_long = pd.merge(df_racer[['race_key', 'lane_no', 'reg_no']], 
                        df_course[['race_key', 'lane_no', 'course']], 
                        on=['race_key', 'lane_no'])
    
    # --- ▼▼▼ ここから修正 ▼▼▼ ---
    # 結合キーに欠損値がある行を削除
    tf2_long.dropna(subset=['reg_no', 'course'], inplace=True)

    # 結合キーのデータ型を整数に変換して統一する
    tf2_long['reg_no'] = tf2_long['reg_no'].astype(int)
    tf2_long['course'] = tf2_long['course'].astype(int)
    # --- ▲▲▲ ここまで修正 ▲▲▲ ---

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
    return df

# --- メイン処理 ---
if __name__ == '__main__':
    # 1. CSVファイルの読み込み
    try:
        # today = datetime.datetime.now()
        # html = download_pred.download_off_pred(today.strftime("%Y%m%d"))
        html_path = 'wakamatsu_beforeinfo_20_20250908_1.html'  # ローカルのHTMLファイルパス
        with open(html_path, 'r', encoding='utf-8') as file:
            html = file.read()
        beforeinfo_df, weather_df = wakamatsu_off_beforeinfo_html_to_csv.parse_boat_race_html(html, is_pred=True)
        print(beforeinfo_df.head())
        print(weather_df.head())
        filtered_course_df = pd.read_csv('filtered_course.csv')
        studium = '若 松'
        yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        race_no = '01'
        beforeinfo_df['race_no'] = race_no
        beforeinfo_df['stadium'] = studium
        beforeinfo_df['race_date'] = yesterday
        weather_df['stadium'] = studium
        weather_df['race_date'] = yesterday
        weather_df['race_no'] = race_no
        filtered_course_df = filtered_course_df[filtered_course_df['stadium'] == studium]
        filtered_course_df.to_csv('filtered_course_若松.csv', index=False)
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
    features_with_record.to_csv('features_with_record.csv', index=False)
    print("\n'features_with_record.csv' として結果を保存しました。")