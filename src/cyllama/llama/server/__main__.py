import argparse
import logging
import time

from .python import ServerConfig, PythonServer


def main():
    parser = argparse.ArgumentParser(description="Llama.cpp Server")
    parser.add_argument("-m", "--model", required=True, help="Path to model file")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--ctx-size", type=int, default=2048, help="Context size")
    parser.add_argument("--gpu-layers", type=int, default=-1, help="GPU layers")
    parser.add_argument("--n-parallel", type=int, default=1, help="Number of parallel processing slots")
    parser.add_argument("--server-type", choices=["python", "embedded"], default="embedded",
                       help="Server implementation to use: python (pure Python) or embedded (high-performance C). Default: embedded")

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

    if args.server_type == "embedded":
        try:
            from .mongoose_server import EmbeddedServer
            print(f"Starting embedded server (high-performance C implementation)")

            server = EmbeddedServer(config)

            if not server.start():
                print("Failed to start embedded server")
                return 1

            try:
                print(f"Embedded server running at http://{args.host}:{args.port}")
                print("Press Ctrl+C to stop...")

                # Run the Mongoose event loop - this blocks until signal received
                server.wait_for_shutdown()
                print("\nShutting down embedded server...")

            finally:
                server.stop()

        except KeyboardInterrupt:
            print("\nReceived KeyboardInterrupt, shutting down...")

        except ImportError:
            print("Embedded server not available. Install with 'make build' to compile Mongoose support.")
            print("Falling back to Python server...")
            args.server_type = "python"

    if args.server_type == "python":
        print(f"Starting Python server")

        with PythonServer(config) as server:
            print(f"Python server running at http://{args.host}:{args.port}")
            print("Press Ctrl+C to stop...")

            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down Python server...")


if __name__ == '__main__':
    main()
