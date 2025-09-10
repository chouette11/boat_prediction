import datetime
import pandas as pd
import download_pred
import wakamatsu_off_beforeinfo_html_to_csv
from pred_features_with_record_from_dfs import build_pred_features_with_record_from_dfs


def on_request_example():
    today = datetime.datetime.now()
    html = download_pred.download_off_pred(today.strftime("%Y%m%d"))
    racers_df, weather_df = wakamatsu_off_beforeinfo_html_to_csv.parse_boat_race_html(html, is_pred=True)
    return racers_df, weather_df
if __name__ == "__main__":
    racers_df, weather_df = on_request_example()
    filtered_course_df = pd.read_csv("filtered_course.csv")
    print(filtered_course_df)
    df = build_pred_features_with_record_from_dfs(
        racers_df=racers_df,
        weather_df=weather_df,
        race_key="2025-09-08-01-20",  # 必須
        race_date="2025-09-08",      # 任意
        filtered_course_df=filtered_course_df,  # 任意
    )
    print(df)
    df.to_csv('test.csv')