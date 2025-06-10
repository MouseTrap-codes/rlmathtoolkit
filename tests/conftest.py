import sys
from pathlib import Path

# Adds the parent directory (your repo root) to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))