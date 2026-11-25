"""最小 HTTP 服务：返回环境变量和工作目录，用来观察 ENV / WORKDIR 的效果。"""
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(os.environ.get("PORT", "8000"))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps(
            {
                "cwd": os.getcwd(),
                "pid": os.getpid(),
                "APP_VERSION": os.environ.get("APP_VERSION"),
                "BUILD_ONLY": os.environ.get("BUILD_ONLY"),
            },
            indent=2,
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print(f"listening on :{PORT}")
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
