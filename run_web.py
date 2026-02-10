#!/usr/bin/env python3
"""Run the win-probability frontend. Requires: data/frontend/ (run run_frontend_data.py first)."""
import sys
from pathlib import Path

# Run from project root so Flask and src resolve
sys.path.insert(0, str(Path(__file__).resolve().parent))

from web.app import app

if __name__ == "__main__":
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True, port=5000)
