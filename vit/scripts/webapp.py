#!/usr/bin/env python3
"""Web 应用启动入口。

Usage:
    python scripts/webapp.py
    python scripts/webapp.py --host 0.0.0.0 --port 8000
    python scripts/webapp.py --config configs/server_config.yaml
"""

import argparse
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(current_dir))

import uvicorn
from server.config import server_config


def main():
    parser = argparse.ArgumentParser(description="Layout Classifier Web Application")
    parser.add_argument("--config", type=str, default="configs/server_config.yaml")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reload", action="store_true", default=False)
    args = parser.parse_args()

    if args.config:
        config_path = args.config
        if not Path(config_path).is_absolute():
            config_path = str((current_dir / config_path).resolve())
        from server.config import ServerConfig
        global server_config
        server_config.__init__(config_path)

    host = args.host or server_config.host
    port = args.port or server_config.port

    print(f"Starting Layout Classifier Platform...")
    print(f"  Host:   {host}")
    print(f"  Port:   {port}")
    print(f"  Config: {server_config.config_path}")
    print(f"  API:    http://{host}:{port}/docs")
    print(f"  Web:    http://{host}:{port}")

    uvicorn.run(
        "server.main:app",
        host=host,
        port=port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
