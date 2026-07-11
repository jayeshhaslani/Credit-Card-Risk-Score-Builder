"""Top-level entry point for the Streamlit app."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps import app as app_module


def main() -> None:
    if hasattr(app_module, "main"):
        app_module.main()
    else:
        app_module.render()


if __name__ == "__main__":
    main()
