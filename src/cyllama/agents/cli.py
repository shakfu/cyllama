#!/usr/bin/env python3
"""
CLI for cyllama agents, including ACP server mode.

Usage:
    # Start ACP server (for editor integration)
    python -m cyllama.agents.cli acp --model path/to/model.gguf

    # Start ACP server with MCP servers
    python -m cyllama.agents.cli acp --model path/to/model.gguf \
        --mcp-stdio "filesystem:npx:-y:@anthropic/mcp-filesystem"

    # Run a simple agent query
    python -m cyllama.agents.cli run --model path/to/model.gguf -p "What is 2+2?"
"""

import argparse
import json
import logging
import os
import sys
from typing import List, Optional

logger = logging.getLogger(__name__)


def parse_mcp_stdio(spec: str) -> dict:
    """
    Parse MCP stdio server specification.

    Format: name:command:arg1:arg2:...
    Example: filesystem:npx:-y:@anthropic/mcp-filesystem
    """
    parts = spec.split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid MCP spec '{spec}': expected name:command[:args...]")

    return {
        "name": parts[0],
        "transport": "stdio",
        "command": parts[1],
        "args": parts[2:] if len(parts) > 2 else [],
    }


def parse_mcp_http(spec: str) -> dict:
    """
    Parse MCP HTTP server specification.

    Format: name:url
    Example: myserver:http://localhost:8080/mcp
    """
    parts = spec.split(":", 1)
    if len(parts) < 2:
        raise ValueError(f"Invalid MCP HTTP spec '{spec}': expected name:url")

    # Reconstruct URL (it contains colons)
    name = parts[0]
    url = spec[len(name) + 1:]

    return {
        "name": name,
        "transport": "http",
        "url": url,
    }


def cmd_acp(args: argparse.Namespace) -> int:
    """Run ACP server."""
    # Import here to avoid circular imports and speed up CLI startup
    from ..api import LLM
    from .acp import ACPAgent
    from .mcp import McpServerConfig, McpTransportType

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,  # Log to stderr, keep stdout for JSON-RPC
    )

    # Load model
    logger.info("Loading model: %s", args.model)
    llm = LLM(args.model)

    # Parse MCP server configurations
    mcp_servers = []

    for spec in args.mcp_stdio or []:
        try:
            config = parse_mcp_stdio(spec)
            mcp_servers.append(McpServerConfig(
                name=config["name"],
                transport=McpTransportType.STDIO,
                command=config["command"],
                args=config["args"],
            ))
        except ValueError as e:
            logger.error("Invalid MCP stdio spec: %s", e)
            return 1

    for spec in args.mcp_http or []:
        try:
            config = parse_mcp_http(spec)
            mcp_servers.append(McpServerConfig(
                name=config["name"],
                transport=McpTransportType.HTTP,
                url=config["url"],
            ))
        except ValueError as e:
            logger.error("Invalid MCP HTTP spec: %s", e)
            return 1

    # Create and start ACP agent
    agent = ACPAgent(
        llm=llm,
        mcp_servers=mcp_servers if mcp_servers else None,
        session_storage=args.session_storage,
        session_path=args.session_path,
        system_prompt=args.system_prompt,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
    )

    logger.info("Starting ACP server")
    try:
        agent.serve()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error("ACP server error: %s", e)
        return 1

    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run a single agent query."""
    from ..api import LLM
    from .react import ReActAgent
    from .tools import tool

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level)

    # Load model
    logger.info("Loading model: %s", args.model)
    llm = LLM(args.model)

    # Create basic tools
    tools = []

    if args.enable_shell:
        import subprocess

        @tool
        def shell(command: str) -> str:
            """Execute a shell command and return the output."""
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                output = result.stdout
                if result.stderr:
                    output += f"\nSTDERR: {result.stderr}"
                if result.returncode != 0:
                    output += f"\nReturn code: {result.returncode}"
                return output
            except subprocess.TimeoutExpired:
                return "Error: Command timed out after 60 seconds"
            except Exception as e:
                return f"Error: {e}"

        tools.append(shell)

    # Create agent
    agent = ReActAgent(
        llm=llm,
        tools=tools,
        system_prompt=args.system_prompt,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
    )

    # Run query
    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompt = f.read()

    if not prompt:
        print("Error: No prompt provided. Use -p or -f.", file=sys.stderr)
        return 1

    result = agent.run(prompt)

    if result.success:
        print(result.answer)
    else:
        print(f"Error: {result.error}", file=sys.stderr)
        return 1

    return 0


def cmd_mcp_test(args: argparse.Namespace) -> int:
    """Test MCP server connection."""
    from .mcp import McpClient, McpServerConfig, McpTransportType

    # Configure logging
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # Parse server spec
    if args.stdio:
        config = parse_mcp_stdio(args.stdio)
        server = McpServerConfig(
            name=config["name"],
            transport=McpTransportType.STDIO,
            command=config["command"],
            args=config["args"],
        )
    elif args.http:
        config = parse_mcp_http(args.http)
        server = McpServerConfig(
            name=config["name"],
            transport=McpTransportType.HTTP,
            url=config["url"],
        )
    else:
        print("Error: Specify --stdio or --http", file=sys.stderr)
        return 1

    # Connect and test
    client = McpClient([server])

    try:
        client.connect_all()
        print(f"Connected to MCP server: {server.name}")

        tools = client.get_tools()
        print(f"\nDiscovered {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")

        resources = client.get_resources()
        print(f"\nDiscovered {len(resources)} resources:")
        for res in resources:
            print(f"  - {res.uri}: {res.name}")

        # Test tool call if specified
        if args.call_tool:
            parts = args.call_tool.split(":", 1)
            tool_name = parts[0]
            tool_args = json.loads(parts[1]) if len(parts) > 1 else {}

            print(f"\nCalling tool: {tool_name}")
            print(f"Arguments: {tool_args}")

            result = client.call_tool(f"{server.name}/{tool_name}", tool_args)
            print(f"Result: {result}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        client.disconnect_all()

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="cyllama agents CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ACP server command
    acp_parser = subparsers.add_parser(
        "acp",
        help="Start ACP server for editor integration",
        description="Start an ACP-compliant agent server that communicates via JSON-RPC over stdio.",
    )
    acp_parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to the model file (GGUF format)",
    )
    acp_parser.add_argument(
        "--mcp-stdio",
        action="append",
        metavar="SPEC",
        help="Add MCP server via stdio (format: name:command:arg1:arg2:...)",
    )
    acp_parser.add_argument(
        "--mcp-http",
        action="append",
        metavar="SPEC",
        help="Add MCP server via HTTP (format: name:url)",
    )
    acp_parser.add_argument(
        "--session-storage",
        choices=["memory", "file", "sqlite"],
        default="memory",
        help="Session storage type (default: memory)",
    )
    acp_parser.add_argument(
        "--session-path",
        help="Path for file/sqlite session storage",
    )
    acp_parser.add_argument(
        "--system-prompt",
        help="Custom system prompt",
    )
    acp_parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum agent iterations (default: 10)",
    )
    acp_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    acp_parser.set_defaults(func=cmd_acp)

    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run a single agent query",
        description="Run a ReAct agent with a single query.",
    )
    run_parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to the model file (GGUF format)",
    )
    run_parser.add_argument(
        "-p", "--prompt",
        help="Prompt to run",
    )
    run_parser.add_argument(
        "-f", "--prompt-file",
        help="File containing the prompt",
    )
    run_parser.add_argument(
        "--system-prompt",
        help="Custom system prompt",
    )
    run_parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum agent iterations (default: 10)",
    )
    run_parser.add_argument(
        "--enable-shell",
        action="store_true",
        help="Enable shell command tool",
    )
    run_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    run_parser.set_defaults(func=cmd_run)

    # MCP test command
    mcp_parser = subparsers.add_parser(
        "mcp-test",
        help="Test MCP server connection",
        description="Connect to an MCP server and list its capabilities.",
    )
    mcp_parser.add_argument(
        "--stdio",
        metavar="SPEC",
        help="MCP server via stdio (format: name:command:arg1:arg2:...)",
    )
    mcp_parser.add_argument(
        "--http",
        metavar="SPEC",
        help="MCP server via HTTP (format: name:url)",
    )
    mcp_parser.add_argument(
        "--call-tool",
        metavar="SPEC",
        help="Call a tool (format: tool_name:json_args)",
    )
    mcp_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    mcp_parser.set_defaults(func=cmd_mcp_test)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
