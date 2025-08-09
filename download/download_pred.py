import os
import datetime
from download_off import download_off
from wakamatsu_off_beforeinfo_html_to_csv import parse_boat_race_html as parse_beforeinfo_html
from wakamatsu_off_racelist_html_to_csv import main as parse_racelist_html
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from sql.etl import predict_main

today = datetime.datetime.now()
kinds = ['racelist', 'beforeinfo']
for kind in kinds:
    print(f"Downloading {kind} for today: {today.strftime('%Y%m%d')}")
    download_off(today.strftime("%Y%m%d"), days=1, interval_sec=1, kind=kind, is_pred=True)

for beforeinfo_filename in os.listdir('download/wakamatsu_off_beforeinfo_pred_html'):
    beforeinfo_path = os.path.join('download/wakamatsu_off_beforeinfo_pred_html', beforeinfo_filename)
    parse_beforeinfo_html(beforeinfo_path, is_pred=True)

parse_racelist_html(is_pred=True)

predict_main()