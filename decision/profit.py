
from dataclasses import dataclass
import pandas as pd


@dataclass
class FarmParams:
    area_ha: float
    yield_t_per_ha: float
    price_unit: str
    costs_total: float


def revenue(price: float, params: FarmParams) -> float:
    total_tons = params.area_ha * params.yield_t_per_ha
    if params.price_unit == 'per_kg':
        return price * total_tons * 1000.0
    return price * total_tons


def profit(price: float, params: FarmParams) -> float:
    return revenue(price, params) - params.costs_total
