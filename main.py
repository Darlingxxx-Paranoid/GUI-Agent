"""
GUI Agent 入口模块
配置 logging，解析命令行参数，启动 AgentLoop
"""
import argparse
import logging
import os
import shutil
import sys

from config import AgentConfig
from agent_loop import AgentLoop


def clean_data_directory(data_dir: str):
    """
    清空 data 目录中的所有内容，避免跨次实验数据污染。
    """
    os.makedirs(data_dir, exist_ok=True)
    for entry in os.listdir(data_dir):
        path = os.path.join(data_dir, entry)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.unlink(path)


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    配置全局 logging
    - 控制台输出（带颜色）
    - 文件输出（可选）
    """
    log_format = (
        "%(asctime)s | %(levelname)-7s | %(name)-30s | %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    level = getattr(logging, log_level.upper(), logging.INFO)

    handlers = []

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    handlers.append(console_handler)

    # 文件 handler（可选）
    if log_file:
        # 如果传入的是目录，在目录下创建默认日志文件
        if os.path.isdir(log_file):
            log_file = os.path.join(log_file, "agent.log")
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别
        file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.DEBUG,  # root logger 设为 DEBUG，由 handler 控制过滤
        handlers=handlers,
        force=True,
    )

    # 降低第三方库的日志级别
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("paddleocr").setLevel(logging.WARNING)
    logging.getLogger("ppocr").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(
        description="基于 Oracle 反馈驱动的 GUI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py --task "打开微信，搜索张三并发送你好"
  python main.py --task "打开设置，连接WiFi" --serial emulator-5554
  python main.py --task "测试" --dry-run --log-level DEBUG
        """,
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        required=True,
        help="任务描述（自然语言）",
    )
    parser.add_argument(
        "--serial", "-s",
        type=str,
        default="",
        help="ADB 设备序列号（默认使用第一个设备）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="干跑模式: 不实际执行 ADB 命令，仅验证流程",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别（默认: INFO）",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="日志输出文件路径（默认仅控制台输出）",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="单个任务最大执行步数（默认: 30）",
    )
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    clean_data_directory(data_dir)

    # 设置 logging
    log_file = args.log_file or os.path.join(
        data_dir, "logs", "agent.log"
    )
    setup_logging(log_level=args.log_level, log_file=log_file)

    logger = logging.getLogger(__name__)
    logger.info("GUI Agent 启动")
    logger.info("任务: %s", args.task)
    logger.info("设备: %s", args.serial or "默认")
    logger.info("Dry Run: %s", args.dry_run)

    # 初始化配置
    config = AgentConfig()
    config.adb_serial = args.serial

    if args.max_steps:
        config.max_steps = args.max_steps

    # 启动 Agent
    agent = AgentLoop(config)
    success = agent.run(task=args.task, dry_run=args.dry_run)

    if success:
        logger.info("🎉 任务执行成功!")
        sys.exit(0)
    else:
        logger.warning("任务未成功完成")
        sys.exit(1)


if __name__ == "__main__":
    main()
