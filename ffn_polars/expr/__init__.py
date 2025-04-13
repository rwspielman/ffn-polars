from .returns import (
    to_returns,
    to_log_returns,
    calc_mtd,
    calc_ytd,
    calc_cagr,
    to_excess_returns,
    rebase,
    to_price_index,
    calc_total_return,
)
from .ratios import (
    calc_sharpe,
    calc_risk_return_ratio,
    calc_information_ratio,
    calc_calmar_ratio,
    sortino_ratio,
    calc_prob_mom,
)
from .risk import (
    calc_max_drawdown,
    ulcer_index,
    ulcer_performance_index,
    to_drawdown_series,
)
from .temporal import (
    year_frac,
    annualize,
    deannualize,
    infer_nperiods,
    infer_freq,
)


expr_funcs = {
    "to_returns": to_returns,
    "to_log_returns": to_log_returns,
    "to_price_index": to_price_index,
    "rebase": rebase,
    "to_drawdown_series": to_drawdown_series,
    "calc_mtd": calc_mtd,
    "calc_ytd": calc_ytd,
    "calc_max_drawdown": calc_max_drawdown,
    "year_frac": year_frac,
    "calc_cagr": calc_cagr,
    "infer_freq": infer_freq,
    "annualize": annualize,
    "deannualize": deannualize,
    "to_excess_returns": to_excess_returns,
    "calc_sharpe": calc_sharpe,
    "calc_risk_return_ratio": calc_risk_return_ratio,
    "calc_information_ratio": calc_information_ratio,
    "calc_prob_mom": calc_prob_mom,
    "calc_total_return": calc_total_return,
    "infer_nperiods": infer_nperiods,
    "sortino_ratio": sortino_ratio,
    "calc_calmar_ratio": calc_calmar_ratio,
    "ulcer_index": ulcer_index,
    "ulcer_performance_index": ulcer_performance_index,
    "calc_ulcer_index": ulcer_index,
    "calc_ulcer_performance_index": ulcer_performance_index,
}
