import yaml

from kvcanary.types import CellSpec


def load_cells(path: str) -> list[CellSpec]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cells = []
    for model in cfg["models"]:
        for comp in cfg["compressors"]:
            for task in cfg["tasks"]:
                cells.append(
                    CellSpec(
                        model=model,
                        method=comp["method"],
                        budget=float(comp["budget"]),
                        task=task,
                    )
                )
    return cells
