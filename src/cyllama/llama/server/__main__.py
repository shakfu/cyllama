import argparse
import logging
import time
import sys

from .embedded import ServerConfig, EmbeddedLlamaServer


def main():
    parser = argparse.ArgumentParser(description="Llama.cpp Server")
    parser.add_argument("-m", "--model", required=True, help="Path to model file")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--ctx-size", type=int, default=2048, help="Context size")
    parser.add_argument("--gpu-layers", type=int, default=-1, help="GPU layers")
    parser.add_argument("--n-parallel", type=int, default=1, help="Number of parallel processing slots")
    parser.add_argument("--server-type", choices=["embedded", "mongoose"], default="embedded",
                       help="Server implementation to use (default: embedded)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = ServerConfig(
        model_path=args.model,
        host=args.host,
        port=args.port,
        n_ctx=args.ctx_size,
        n_gpu_layers=args.gpu_layers,
        n_parallel=args.n_parallel
    )

    if args.server_type == "mongoose":
        try:
            from .mongoose_server import MongooseServer
            print(f"Starting Mongoose server (high-performance C implementation)")

            server = MongooseServer(config)

            if not server.start():
                print("Failed to start Mongoose server")
                sys.exit(1)

            print(f"Mongoose server running at http://{args.host}:{args.port}")
            print("Press Ctrl+C to stop...")

            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down Mongoose server...")
                server.stop()

        except ImportError:
            print("Mongoose server not available. Install with 'make build' to compile Mongoose support.")
            print("Falling back to embedded Python server...")
            args.server_type = "embedded"

    if args.server_type == "embedded":
        print(f"Starting embedded Python server")

        with EmbeddedLlamaServer(config) as server:
            print(f"Embedded server running at http://{args.host}:{args.port}")
            print("Press Ctrl+C to stop...")

            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down embedded server...")


if __name__ == '__main__':
    main()
