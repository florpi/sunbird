import argparse
import yaml
import time
from pathlib import Path
from sunbird.inference import Nested

if __name__ == "__main__":
    output_path = Path("/n/home11/ccuestalazaro/sunbird/scripts/inference/chains/")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, default="configs/infer_combined.yaml"
    )
    # Make sure it reads from dataset with fixed hod
    parser.add_argument("--cosmology", type=int, default=0)
    parser.add_argument("--hod_idx", type=int, default=940)
    parser.add_argument("--suffix", type=str, default=None)
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    config["data"]["observation"]["args"]["cosmology"] = args.cosmology
    config["data"]["observation"]["args"]["hod_idx"] = args.hod_idx
    dir_store = f"abacus_cosmo{args.cosmology}_hod{args.hod_idx}"
    if args.suffix is not None:
        dir_store += f"_{args.suffix}"
    config["inference"]["output_dir"] = output_path / dir_store
    print("output dir")
    print(config["inference"]["output_dir"])
    nested = Nested.from_config_dict(
        config=config,
    )
    t0 = time.time()
    print(f"Fitting parameters {nested.param_names}")
    nested()
    print("Fitting took = ", time.time() - t0)
