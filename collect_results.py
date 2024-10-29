import shutil
from pathlib import Path

result_dir = Path(__file__).parent / "results"
if not result_dir.exists():
    result_dir.mkdir()

# collect *.csv
for item in Path(__file__).parent.iterdir():
    if not item.is_dir():
        continue
    if item.name in ["results", "dataset"]:
        continue
    for file in item.iterdir():
        if file.suffix == ".csv":
            shutil.copy(file, result_dir / file.name)
        if "result" in file.name and file.suffix == ".json":
            shutil.copy(file, result_dir / file.name)
        if "result" in file.name and file.suffix == ".png":
            shutil.copy(file, result_dir / file.name)
