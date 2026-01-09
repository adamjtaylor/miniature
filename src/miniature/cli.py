"""Command-line interface for miniature."""

import sys


def main():
    """Main entry point for miniature CLI."""
    from .core import main as core_main
    core_main()


def metrics_main():
    """Entry point for miniature-metrics CLI."""
    from .metrics import main as metrics_main_impl
    metrics_main_impl()


if __name__ == "__main__":
    main()
