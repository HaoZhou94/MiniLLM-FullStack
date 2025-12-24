import logging
import os
from datetime import datetime


def setup_logger():

    """
        配置项目日志系统：
        1. 日志级别为INFO
        2. 同时输出到控制台和文件（outputs/logs/）
        3. 日志格式包含时间、模块、级别、消息
    """
    # 创建日志目录
    # log_dir= "outputs/logs"
    # os.makedirs(log_dir, exist_ok=True)

    log_dir = "outputs/logs"
    os.makedirs(log_dir, exist_ok=True)


    # 日志文件名（按时间命名）
    log_file = os.path.join(log_dir, f"qwen_micro_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # 配置日志器
    logger = logging.getLogger("qwen_micro")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # 清除默认处理器

    # 日志文件名
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    # 日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()

