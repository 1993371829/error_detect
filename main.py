"""
项目入口脚本。

将命令行调用转发至 stage_1.cli，便于直接运行:
    python main.py --input data.csv --dry-run
"""

from stage_1.cli import main

if __name__ == "__main__":
    main()
