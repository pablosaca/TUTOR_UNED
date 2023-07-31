from typing import Union
import pandas as pd

FloatOrIntOrStrOrNone = Union[float, int, str, None]
FloatOrIntOrStr = Union[float, int, str]
FloatOrStrOrDict = Union[float, str, dict]
FloatOrStr = Union[float, str]
FloatOrDictOrNone = Union[float, dict, None]
FloatOrStrOrDictOrNone = Union[float, str, dict, None]
FloatOrNone = Union[float, None]
FloatOrIntOrNone = Union[float, int, None]
FloatOrInt = Union[float, int]
StrOrNone = Union[str, None]
IntOrNone = Union[int, None]
DictOrNone = Union[dict, None]
DataFrameOrNone = Union[None, pd.DataFrame]
DataFrameOrSeries = Union[pd.Series, pd.DataFrame]
ListOrNone = Union[list, None]


class Error(Exception):
    """Base class for other exceptions"""
    pass
