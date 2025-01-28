from snakemake.script import snakemake
import pandas as pd
from pathlib import Path
import json
from typing import Optional


def main(
    input_paths: list[Path],
    output_path: Path,
    added_columns: Optional[dict] = None,
    ignore_keys: Optional[list[str]] = None,
):
    output_path = Path(output_path)
    data = dict()

    for idx, ip in enumerate(input_paths):
        with open(ip, "r") as f:
            res = json.load(f)

        res["file"] = str(ip)

        if not added_columns is None:
            for k in added_columns.keys():
                item = added_columns[k][idx]
                if k in data:
                    data[k].append(item)
                else:
                    data[k] = [item]

        for k, v in res.items():
            if not ignore_keys is None and k in ignore_keys:
                continue
            if k in data:
                data[k].append(v)
            else:
                data[k] = [v]

    df = pd.DataFrame(data)
    with open(output_path, "w") as f:
        if output_path.suffix == ".csv":
            df.to_csv(f)
        elif output_path.suffix == ".json":
            df.to_json(f, indent=4)
        elif output_path.suffix == ".hdf5":
            df.to_hdf(f, key="data")
        else:
            raise Exception(f"{output_path.suffix} is not a valid file extension")


if __name__ == "__main__":
    main(
        snakemake.input,
        snakemake.output[0],
        snakemake.params.get("add_columns", None),
        snakemake.params.get("ignore_columns", None),
    )
