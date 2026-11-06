import requests
import yaml

VERSION = 1  # 改这个数字来模拟"修改源码"

if __name__ == "__main__":
    print("version", VERSION, "requests", requests.__version__, "yaml", yaml.__version__)
