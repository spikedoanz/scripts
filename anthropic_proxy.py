#!/usr/bin/env python3
"""
LiteLLM proxy server for Anthropic models with OpenAI-compatible API.

Usage:
    python scripts/anthropic_proxy.py --model claude-3-5-sonnet-20241022

    # Custom port
    python scripts/anthropic_proxy.py --model claude-3-5-haiku-20241022 --port 8001

    # Multiple models
    python scripts/anthropic_proxy.py --config scripts/litellm_config.yaml

Environment variables:
    ANTHROPIC_API_KEY: Your Anthropic API key (required)
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def create_config_file(models: list[str], output_path: Path) -> Path:
    """Create a LiteLLM config file for multiple Anthropic models."""
    config_content = "model_list:\n"

    for model in models:
        # LiteLLM expects anthropic/ prefix for Anthropic models
        litellm_model = f"anthropic/{model}" if not model.startswith("anthropic/") else model
        config_content += f"""
  - model_name: {model}
    litellm_params:
      model: {litellm_model}
      api_key: os.environ/ANTHROPIC_API_KEY
"""

    output_path.write_text(config_content)
    print(f"✓ Created config file: {output_path}")
    return output_path


def start_proxy(
    model: str | None = None,
    config: Path | None = None,
    port: int = 4000,
    host: str = "0.0.0.0",
):
    """Start the LiteLLM proxy server."""

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set")
        print("\nSet it with:")
        print("  export ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)

    # Build command
    cmd = ["litellm", "--port", str(port), "--host", host]

    if config:
        cmd.extend(["--config", str(config)])
    elif model:
        # Single model mode
        litellm_model = f"anthropic/{model}" if not model.startswith("anthropic/") else model
        cmd.extend(["--model", litellm_model])
    else:
        print("❌ Error: Must specify either --model or --config")
        sys.exit(1)

    # Drop params that Anthropic doesn't support but OpenAI clients might send
    cmd.append("--drop_params")

    print(f"🚀 Starting LiteLLM proxy server...")
    print(f"   Model(s): {model if model else 'from config'}")
    print(f"   Endpoint: http://{host}:{port}")
    print(f"   OpenAI-compatible: http://{host}:{port}/v1")
    print(f"\n📝 Use with verifiers:")
    if model:
        print(f'   client = OpenAI(api_key="dummy", base_url="http://{host}:{port}")')
        print(f'   results = env.evaluate(client=client, model="{model}", ...)')
    print(f"\n   Or add to configs/endpoints.py:")
    print(f'   "{model or "your-model"}": {{')
    print(f'       "model": "{model or "your-model"}",')
    print(f'       "url": "http://{host}:{port}",')
    print(f'       "key": "EMPTY",')
    print(f"   }}")
    print(f"\n{'='*60}")
    print("Press Ctrl+C to stop the proxy\n")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n✓ Proxy server stopped")
    except FileNotFoundError:
        print("\n❌ Error: litellm not found. Install it with:")
        print("   pip install 'litellm[proxy]'")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Start LiteLLM proxy for Anthropic models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model
  python scripts/anthropic_proxy.py --model claude-3-5-sonnet-20241022

  # Multiple models via config
  python scripts/anthropic_proxy.py --models claude-3-5-sonnet-20241022 claude-3-5-haiku-20241022

  # Custom config file
  python scripts/anthropic_proxy.py --config my_config.yaml

  # Custom port
  python scripts/anthropic_proxy.py --model claude-3-5-sonnet-20241022 --port 8001

Available Anthropic models:
  - claude-3-5-sonnet-20241022
  - claude-3-5-haiku-20241022
  - claude-3-7-sonnet-20250219
  - claude-3-opus-20240229
        """
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Single Anthropic model to proxy"
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Multiple Anthropic models (creates temp config)"
    )

    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to LiteLLM config file"
    )

    parser.add_argument(
        "--port", "-p",
        type=int,
        default=4000,
        help="Port to run proxy on (default: 4000)"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )

    args = parser.parse_args()

    # Handle multiple models
    if args.models:
        config_path = Path("scripts/litellm_anthropic_config.yaml")
        create_config_file(args.models, config_path)
        start_proxy(config=config_path, port=args.port, host=args.host)
    else:
        start_proxy(
            model=args.model,
            config=args.config,
            port=args.port,
            host=args.host
        )


if __name__ == "__main__":
    main()
