import argparse
import logging
import time

from .embedded import ServerConfig, EmbeddedLlamaServer


def main():
    parser = argparse.ArgumentParser(description="Embedded Llama.cpp Server")
    parser.add_argument("-m", "--model", required=True, help="Path to model file")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--ctx-size", type=int, default=2048, help="Context size")
    parser.add_argument("--gpu-layers", type=int, default=-1, help="GPU layers")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = ServerConfig(
        model_path=args.model,
        host=args.host,
        port=args.port,
        n_ctx=args.ctx_size,
        n_gpu_layers=args.gpu_layers
    )

    with EmbeddedLlamaServer(config) as server:
        print(f"Server running at http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop...")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down server...")


if __name__ == '__main__':
    main()
