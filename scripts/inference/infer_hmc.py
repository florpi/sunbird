import argparse
import time
from sunbird.inference import HMC 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, default="configs/infer_cross_hmc.yaml"
    )
    args = parser.parse_args()
    nested = HMC.from_config(args.config_path)
    t0 = time.time()
    print(f"Fitting parameters {nested.param_names}")
    nested()
    print("Fitting took = ", time.time() - t0)
