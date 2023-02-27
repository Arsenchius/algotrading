from strategies import *
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(Path(os.path.dirname(SCRIPT_DIR)), "model_training"))
from experiments import EXPERIMENT_ID_TO_FEATURES

EXPERIMENT_ID_TO_STRATEGY = {
    1: SmaCross,
    2: MultipleBasic,
}
