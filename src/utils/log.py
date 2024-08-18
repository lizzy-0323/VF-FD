import logging


class ColoredFilter(logging.Filter):
    def filter(self, record):
        logger_name = record.name
        if logger_name == "client_1":
            record.color = "\033[1;31m"  # 红色
        elif logger_name == "client_2":
            record.color = "\033[1;32m"  # 绿色
        elif logger_name == "client_3":
            record.color = "\033[1;33m"  # 黄色
        elif logger_name == "client_4":
            record.color = "\033[1;34m"  # 蓝色
        else:
            record.color = "\033[0m"  # 默认颜色
        return True

def init_logger(name, log_path):
    """
    init logger
    :param name: logger name
    :param log_path: log path
    :return: logger
    """
    # 清除已有的log
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(color)s%(asctime)s - %(levelname)s - %(message)s\033[0m"
        )
        # 添加彩色输出
        colored_filter = ColoredFilter()
        logger.addFilter(colored_filter)
        fh = logging.FileHandler(log_path, mode="w")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger


if __name__ == "__main__":
    logger = init_logger("test", "test.log")
    logger.info("test")
