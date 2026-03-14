import pandas as pd
import requests
from io import BytesIO

BASE_URL = "https://pub-3723ecad7f3943b4b6b29ffeb24bb0fb.r2.dev"
UR_NEW_URL = f"{BASE_URL}/ur_yosou.parquet"
UR_URL = f"{BASE_URL}/ur_bunseki.parquet"

def read_parquet_from_url(url: str) -> pd.DataFrame:
    r = requests.get(url)
    r.raise_for_status()
    return pd.read_parquet(BytesIO(r.content), engine="pyarrow")

def load_data():
    ur_new = read_parquet_from_url(UR_NEW_URL)
    ur = read_parquet_from_url(UR_URL)

    ur_pre = ur.merge(
        ur_new[["レースid","開催日","クラス","コース","R","芝砂","距離","馬名","騎手","馬番","枠番","距離区分"]],
        on="馬名",
        how="left",
        suffixes=("", "_now")
    )

    ur_pre["開催日"] = pd.to_datetime(ur_pre["開催日"], errors="coerce")

    return ur, ur_new, ur_pre