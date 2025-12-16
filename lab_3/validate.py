import re
import pandas as pd


def validate(data: pd.DataFrame, regular_expressions: dict[str, str]) -> list:
    """
    Validates DataFrame with regular expressions

    :param data: DataFrame
    :param regular_expressions: Regular expressions from file
    :return: list of wrong rows numbers
    """
    results = set()
    for col, expr in regular_expressions.items():
        for i in data.index:
            if not bool(re.fullmatch(expr, str(data.loc[i, col]))):
                results.add(i)

    return sorted(results)
