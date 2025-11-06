"""Generate the OpenAPI schema for the scoring API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from api import app as api_app


def generate_openapi(output_path: Path) -> None:
    """Render the FastAPI OpenAPI schema to the provided path."""

    schema = api_app.app.openapi()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"OpenAPI spec written to {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate OpenAPI schema for the scoring API"
    )
    parser.add_argument(
        "--output",
        default="docs/openapi/openapi.json",
        help="Destination path for the OpenAPI JSON document",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    generate_openapi(output_path)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
