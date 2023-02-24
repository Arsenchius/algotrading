import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import argparse
import json

from multiprocessing import Process
import pandas as pd
from typing import NoReturn, List
from dataset import *
from model import *
from experiments import EXPERIMENT_ID_TO_FEATURES
from tqdm import tqdm_notebook

def _run_part(
    output_path: str,
    exp_id: int,
    model_name: str,
    data: Data,
    task_type: str,
    metric: str,
    is_optimized: bool = False,
    type_of_training: str = "default",
):
    model = Model(model_name=model_name)
    FEATURES = EXPERIMENT_ID_TO_FEATURES[exp_id]
    result = model.train(data, metric=metric, features=FEATURES, is_optimized=is_optimized)

    with open(output_path, 'w') as f:
        f.write(json.dumps({
            'features': FEATURES,
            'metric': metric,
            'result': result,
        }))
        f.write('\n')


def run(
    result_dir_path: str,
    data_path: str,
    expirements: List,
    model_name: str,
    task_type:str,
    metric: str,
    is_optimized: bool = False,
    type_of_training: str = "default",
) -> NoReturn:
    # result_dir_path = args.result_dir_path
    # data_path = args.data_path
    # exp_id = args.exp_id
    # output_dir_path = args.output_dir_path
    # completions_path = args.completions_path
    # model_path = args.model_path

    if not os.path.exists(result_dir_path):
        os.mkdir(result_dir_path)

    part_jobs = []
    df = pd.read_csv(data_path)
    df.set_index(['Time'], inplace=True)
    data = Data(df)
    data.create_features_and_target(task_type=task_type)
    for exp_id in tqdm_notebook(expirements):
        full_part_output_path = os.path.join(result_dir_path, f"res-{exp_id}")

        part_jobs.append(
            Process(
                target=_run_part,
                args=(
                    full_part_output_path,
                    exp_id,
                    model_name,
                    data,
                    task_type,
                    metric,
                    is_optimized,
                    type_of_training,
                ),
            )
        )

    for job in part_jobs:
        job.start()

    for job in part_jobs:
        job.join()



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--result-dir-path', type=str, help='Path to result dir', required=True)
#     parser.add_argument('--data-path', type=str, help='Path to prepared data')
#     # parser.add_argument('--completions-path', type=str, help='Path to prepared prefix->completions mapping', required=True)
#     # parser.add_argument('--model-path', type=str, help='Path to model', required=True)
#     parser.add_argument('--exp-id', type=int, help='Id of experiment', required=True)
#     args = parser.parse_args()

#     run(args)
