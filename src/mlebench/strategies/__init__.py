"""
Strategy template selector for MLE-Bench competitions.
"""

from mlebench.strategies.tabular import TABULAR_TEMPLATE
from mlebench.strategies.nlp import NLP_TEMPLATE
from mlebench.strategies.vision_ml import VISION_TEMPLATE
from mlebench.strategies.timeseries import TIMESERIES_TEMPLATE
from mlebench.strategies.general import GENERAL_TEMPLATE


STRATEGY_MAP = {
    "tabular": TABULAR_TEMPLATE,
    "nlp": NLP_TEMPLATE,
    "vision": VISION_TEMPLATE,
    "timeseries": TIMESERIES_TEMPLATE,
    "general": GENERAL_TEMPLATE,
}


def get_strategy_template(strategy: str) -> str:
    """Get the strategy template code for a given strategy name."""
    return STRATEGY_MAP.get(strategy, GENERAL_TEMPLATE)
