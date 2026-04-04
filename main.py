#!/usr/bin/env python3
"""
main.py - Entry point for the Personal Knowledge Base system.
Usage:
    python main.py init                    # Initialize KB
    python main.py add <path>              # Add file or directory
    python main.py search <query>          # Search
    python main.py serve                   # Start web UI
    python main.py --help                  # Full help
"""

import sys
import os

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.cli import main

if __name__ == "__main__":
    main()
