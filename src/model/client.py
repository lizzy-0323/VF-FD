"""
@Author: laziyu
@Date:2023-1-24
@Description: client node
"""

from utils import distance
from utils.const import IP, PER_ENCRYPT_TIME, QUERY_NUM, ROW_NUM, K


class Client:
    def __init__(self, silo, distance_metric) -> None:
        """
        :param silo: dataframe
        :param distance_metric: distance metric
        """
        self.silo = silo
        self.distance_metric = distance_metric
        self.ids = self._gen_id_list()

    def _get_cols_intersection(self, query):
        """
        :param query: dataframe
        :return: columns intersection between query and silo
            silo: dataframe
            query: dataframe
        """
        # 通过相交的列重新确定新的query和data
        intersect_cols = self.silo.columns.intersection(query.columns)
        base = self.silo[intersect_cols]
        query = query[intersect_cols]
        return base, query

    def _query(self, query):
        """
        client local query
        :param query: query
        :return:
            id_list
            query_result
        """

        data, query = self._get_cols_intersection(query)
        k = data.shape[0]
        distances = distance(data.values, query.values, metric=self.distance_metric)
        data["distance"] = distances
        result = data.sort_values(
            by="distance",
            ascending=self.distance_metric == "l2",
        )
        index_list = result.index[:k].tolist()
        return index_list, result

    def _gen_id_list(self):
        """
        gen id list
        """
        return self.silo.index.tolist()

    def _update_index(self, id_list):
        """
        update index
        :param id_list: id list
        """
        try:
            self.silo = self.silo.loc[id_list]
            self.ids = id_list
        except:
            raise ValueError("index not exist in data")
