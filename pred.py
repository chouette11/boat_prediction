import os
import datetime
from download.download_pred import download_off_pred
from download.wakamatsu_off_beforeinfo_html_to_csv import parse_boat_race_html as parse_beforeinfo_html
import sql2.etl_pred

today = datetime.datetime.now()
yesterday = today - datetime.timedelta(days=1)
print(f"Downloading yesterday: {yesterday.strftime('%Y%m%d')}")
download_off_pred(today.strftime("%Y%m%d"))

for beforeinfo_filename in os.listdir('download/wakamatsu_off_beforeinfo_pred_html'):
    beforeinfo_path = os.path.join('download/wakamatsu_off_beforeinfo_pred_html', beforeinfo_filename)
    parse_beforeinfo_html(beforeinfo_path, is_pred=True)


sql2.etl_pred.predict_main()