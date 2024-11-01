"""
@Description: distance operations
@Author: laziyu
@Date: 2023-1-13
"""

import numpy as np
from scipy.stats import wasserstein_distance


def euclidean_distance(data, query):
    """
    calculate euclidean distance between data and query
    :param data: data
    :param query: query
    :return: distance
    """
    return np.linalg.norm(data - query, axis=1)


def earth_mover_distance(arr1, arr2):
    """
    calculate earth mover distance
    """
    return wasserstein_distance(arr1, arr2)


def cosine_distance(data, query):
    """
    calculate cosine distance between data and query
    :param data: data
    :param query: query
    :return: distance
    """
    return np.dot(data, query.T) / (np.linalg.norm(data) * np.linalg.norm(query))


def distance(data, query, metric="cosine"):
    """
    calculate distance between data and query
    :param data: data
    :param query: query
    :param type: distance type
    :return: distance
    """
    if metric == "l2":
        return euclidean_distance(data, query)
    if metric == "cosine":
        return cosine_distance(data, query)
    if metric == "emd":
        return earth_mover_distance(data, query)
    raise ValueError("distance type must be l2 or cosine")


def get_dist_by_row(silos, row):
    """
    get distance of row
    :param silos: silo list
    :param row: row index
    :return: distance list
    """
    distances = []
    for silo in silos:
        distances.append(silo.iloc[row]["distance"])
    return distances


def get_dist_by_id(silos, idx):
    """
    get distance by id
    :param silos: silo list
    :param id: id
    :return: distance list
    """
    distances = []
    for silo in silos:
        distances.append(silo.loc[idx, "distance"])
    return distances


def sum_all_distance(silos, type="multiple"):
    """
    sum all distance in silos
    :param silos: silo list
    :return: sum of all distance
    """
    # 合并每一个silo的distance到第一个，并删除其他的silo的distance
    if type == "multiple":
        for i in range(len(silos) - 1):
            silos[-1]["distance"] *= silos[i]["distance"]
        for silo in silos[:-1]:
            del silo["distance"]

        return silos
    if type == "add":
        for silo in silos:
            silo["distance"] = silo["distance"].map(lambda x: x**2)
        for i in range(len(silos) - 1):
            silos[-1]["distance"] += silos[i]["distance"]
        silos[-1]["distance"] = silos[-1]["distance"].map(lambda x: np.sqrt(x))
        for silo in silos[:-1]:
            del silo["distance"]
        return silos
    raise ValueError("type must be multiple or add")
