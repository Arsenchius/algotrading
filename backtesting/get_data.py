import os
import pandas as pd
from tqdm import tqdm
from binance import Client
from sqlalchemy import create_engine
from typing import NoReturn, List
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
