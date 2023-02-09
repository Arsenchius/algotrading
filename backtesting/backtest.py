import os
import config
import pandas as pd
from tqdm import tqdm
from datetime import timedelta
from sqlalchemy import create_engine, text
from backtesting import Backtest, Strategy


def backtest(
    strategy: type[Strategy],
    output_dir_path: str,
    url: str,
    time_frame: str,
    td_days: int,
    cash: float = 10000,
    commission: float = 0.001,
) -> pd.DataFrame():
    engine = create_engine(url=url)
    returns = []
    # outputs = []

    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)

    full_dir_output_path = os.path.join(output_dir_path, f"res_{time_frame}/")

    if not os.path.exists(full_dir_output_path):
        os.mkdir(full_dir_output_path)

    with engine.connect() as connection:
        query = "SELECT name FROM sqlite_schema WHERE type='table'"
        symbols = [item[0] for item in connection.execute(text(query)).fetchall()]
        for symbol in symbols:
            qry = f"SELECT * FROM '{symbol}' WHERE Open_Time < '{pd.to_datetime('today') - timedelta(days = td_days)}'"
            data = pd.DataFrame(connection.execute(text(qry))).set_index("Open_Time")
            data.index = pd.to_datetime(data.index)

            bt = Backtest(
                data, strategy, cash=cash, commission=commission, exclusive_orders=True
            )
            output = bt.run()
            # outputs.append(output)
            output = pd.concat(
                [pd.Series([symbol, time_frame], ["Pair", "TimeFrame"]), output]
            )
            full_part_output_path = os.path.join(
                full_dir_output_path, f"res_{symbol}.json"
            )
            output.to_json(full_part_output_path, date_format="iso")
            returns.append(output["Return [%]"])
            # bt.plot(filename=f"{symbol}_result.html",open_browser=False)

    connection.close()

    df = pd.DataFrame(returns, index=symbols, columns=["return_%"])
    df = df.rename_axis("pair")
    top_5 = df.nlargest(5, columns=["return_%"])
    top_5.to_json(os.path.join(full_dir_output_path, "top_5.json"))

    return top_5


def validate():
    pass
