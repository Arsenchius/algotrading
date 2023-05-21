import os
import sys
from typing import NoReturn, List
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from binance import Client
from sqlalchemy import create_engine


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(Path(os.path.dirname(SCRIPT_DIR)), "configs"))

from config import *


def get_data(pairs: List[str], start_date: str, interval: str, url: str) -> NoReturn:
    engine = create_engine(url=url)
    client = Client(API_KEY, SECRET_KEY)
    for pair in tqdm(pairs):
        df = pd.DataFrame(client.get_historical_klines(pair, interval, start_date))
        df.columns = [
            "Open_Time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Close_Time",
            "Quote_Asset_Volume",
            "Number_of_Trades",
            "TB_Base_Volume",
            "TB_Quote_Volume",
            "Ignore",
        ]
        df.Open_Time = pd.to_datetime(df.Open_Time, unit="ms")
        df.Close_Time = pd.to_datetime(df.Close_Time, unit="ms")
        df.set_index("Open_Time", inplace=True)
        df = df.drop(
            columns=[
                "Close_Time",
                "Quote_Asset_Volume",
                "Number_of_Trades",
                "TB_Base_Volume",
                "TB_Quote_Volume",
                "Ignore",
            ]
        )
        df[
            [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]
        ] = df[
            [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]
        ].astype(float)
        df.to_sql(pair, engine, if_exists="replace")
