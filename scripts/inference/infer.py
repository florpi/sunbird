from sunbird.inference import Nested
import time

if __name__ == "__main__":
    nested = Nested.from_config("configs/infer_large.yaml")
    t0 = time.time()
    print(f"Fitting parameters {nested.param_names}")
    nested(log_dir="chains/", slice_steps=100, num_live_points=400)
    print("Fitting took = ", time.time() - t0)
