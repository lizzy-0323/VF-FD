from utils.log import init_logger


def test_logger():
    logger = init_logger("leader", log_path="./log/test.txt")
    logger.info("leader starts")


if __name__ == "__main__":
    test_logger()
