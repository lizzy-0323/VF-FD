"""
@Author: laziyu
@Date:2023-2-14
@Description: change dataframe type to protobuf 
"""

import pandas as pd

from grpc_service.rpc import basic_pb2


def protobuf_to_df(df_protobuf):
    """
    protobuf to dataframe
    :param: df_protobuf: dataframe in protobuf
    :return: dataframe
    """
    df_format_data = pd.DataFrame()
    # 遍历DataFrame消息中的Series
    for series_msg in df_protobuf.series:
        # 创建一个新的Series对象
        series_data = {}
        # 遍历每个Series对象中的列
        for column_name, value_msg in series_msg.columns.items():
            # 根据值的类型添加到Series中
            if value_msg.HasField("int_value"):
                series_data[column_name] = value_msg.int_value
            elif value_msg.HasField("float_value"):
                series_data[column_name] = value_msg.float_value
            elif value_msg.HasField("string_value"):
                series_data[column_name] = value_msg.string_value
            else:
                raise TypeError("type not implement")
        # 获取当前行的索引
        index_value = series_msg.index
        # 创建带有索引的Series对象
        series = pd.Series(series_data, name=index_value)
        # 将Series添加到DataFrame中
        df_format_data = pd.concat([df_format_data, series.to_frame().T])
    return df_format_data


def df_to_protobuf(df):
    """
    dataframe to protobuf dataframe
    :param: df: dataframe
    :return: protobuf dataframe
    """
    protobuf_df = basic_pb2.dataframe()
    for index, row in df.iterrows():
        series_msg = protobuf_df.series.add()
        series_msg.index = index
        # 遍历Series的列，并将每个值添加到行对象中
        for column_name, value in row.items():
            # 创建一个新的值对象
            value_msg = series_msg.columns[column_name]
            # 根据值的类型设置相应的字段
            if isinstance(value, int):
                value_msg.int_value = value
            elif isinstance(value, float):
                value_msg.float_value = value
            elif isinstance(value, str):
                value_msg.string_value = value
            else:
                raise TypeError("type not implement")
    return protobuf_df


def series_to_protobuf(series):
    """
    series to protobuf series
    :param: series: series
    :return: protobuf series
    """
    protobuf_series = basic_pb2.series()
    row_msg = protobuf_series.row
    # 遍历Series的索引标签和值，并将其添加到行对象中
    for label, value in series.items():
        # 创建一个新的值对象
        value_msg = row_msg.columns[label]
        # 根据值的类型设置相应的字段
        if isinstance(value, int):
            value_msg.int_value = value
        elif isinstance(value, float):
            value_msg.float_value = value
        elif isinstance(value, str):
            value_msg.string_value = value
        else:
            raise TypeError("type not implemented")

    return protobuf_series
