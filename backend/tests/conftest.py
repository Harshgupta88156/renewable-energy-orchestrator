import os
import sys
from pathlib import Path

os.environ.setdefault("REO_PERSIST", "0")  # tests never touch the real database
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
