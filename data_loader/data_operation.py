"""
@Description: data operations
@Author: laziyu
@Date: 2023-1-25
"""

import numpy as np
import pandas as pd


def align(silos):
    """
    Align silo by first silo
    :param silo: silo list
    :return aligned silos
    """
    # 按照第一个silo的id对齐
    n = len(silos)
    for i in range(n - 1):
        silos[i] = silos[i].reindex(index=silos[0].index)
    for i in range(n):
        silos[i] = silos[i].fillna(0)
    return silos


def get_row_intersection(silos):
    """
    get row intersection
    :param silos: silo list
    :return: row intersection
    """
    index = silos[0].index
    n = len(silos)
    for i in range(n):
        index = index.intersection(silos[i].index)
    return [silo.loc[index] for silo in silos]


def get_row_union(silos):
    """
    get row union
    :param silos: silo list
    :return: row union
    """
    index = silos[0].index
    n = len(silos)
    for i in range(n):
        index = index.union(silos[i].index)
    return [silo.loc[index] for silo in silos]


def rand_select_col(df, num, random_state=0):
    """
    sample column from silo
    :param df: df
    :param num: number of columns
    :return selected columns
    """
    return df.sample(num, axis=1, random_state=random_state)


def merge(*silos, ignore_index=False, axis=1):
    """
    merge multiple silos into one, index is the first silo's index
    :param silos: silo list
    :param ignore_index: whether to ignore index, False:-> simple concat
    :return: merged silo
    """
    if not ignore_index:
        return pd.concat(silos, axis=axis, copy=True)
    merged_silos = None
    for silo in silos:
        merged_silos = (
            np.concatenate([merged_silos, silo.values], axis=1)
            if merged_silos is not None
            else silo.copy().values
        )
    return pd.DataFrame(
        merged_silos,
        columns=[i for silo in silos for i in silo.columns],
        index=silos[0].index,
    )


def overlapping(columns):
    """
    get overlapping columns
    Two steps:
    1. get intersection of columns
    2. get union of columns
    :param silos: silo list
    :return: overlapping columns
    """
    overlapping_col_name = set()
    n = len(columns)
    for i in range(n):
        for j in range(i + 1, n):
            # 取并集
            temp = set(columns[i]) & set(columns[j])
            overlapping_col_name = overlapping_col_name.union(temp)
    # print(overlapping_cols)
    return overlapping_col_name
