# -----------
# > Imports <
# -----------
from pathlib import Path



# ----------
# > Helper <
# ----------
def save_dir_creation(dir_path:str):
    path = Path(dir_path)
    parts = path.parts  # for example: ['/', 'mnt', 'data_2', 'ippolito', 'checkpoints', ...]

    cur_path = Path(parts[0])
    for cur_part in parts[1:]:
        cur_path = cur_path / cur_part

        # print(f"Creating: {cur_path}")
        # print(f"Exists: {cur_path.exists()}")
        # print(f"Parent: {cur_path.parent.exists()}")

        try:
            cur_path.mkdir(exist_ok=True)
        except PermissionError:
            pass















