import os
import argparse
import json
import csv
import pandas as pd
from multiprocessing import Process
from binance import Client
from datetime import datetime
from typing import NoReturn
from config import *
from get_data import *
from backtest import *
from strategies import *
from experiments import EXPERIMENT_ID_TO_STRATEGY


def _run_part(
    url: str,
    output_dir_path: str,
    strategy_id: int,
    td_days: int,
    start_date: str,
    time_frame: str,
) -> NoReturn:
    get_data(tranding_pairs, start_date, time_frame, url)

    top_5 = backtest(
        EXPERIMENT_ID_TO_STRATEGY[strategy_id],
        output_dir_path,
        url,
        time_frame,
        td_days,
    )




def run(args):
    output_dir_path = args.output_dir_path
    td_days = args.time_delta_days
    strategy_id = args.strategy_id

    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)

    start_date = "30 days ago UTC"
    part_jobs = []

    for time_interval in time_frame_to_data_base:
        part_jobs.append(
            Process(
                target=_run_part,
                args=(
                    time_frame_to_data_base[time_interval],
                    output_dir_path,
                    strategy_id,
                    td_days,
                    start_date,
                    time_interval,
                ),
            )
        )

    for job in part_jobs:
        job.start()

    for job in part_jobs:
        job.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir-path",
        type=str,
        help="Path to output dumped back testing results",
        required=True,
    )
    parser.add_argument(
        "--time-delta-days",
        type=int,
        help="Number of days ago, date until we backtesting",
        required=True
    )
    parser.add_argument(
        "--strategy-id",
        type=int,
        help="Id of strategy",
        required=True,
    )
    args = parser.parse_args()

    run(args)
