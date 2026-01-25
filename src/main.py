from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]

print("Project root:", ROOT)
print("Now:", datetime.now())