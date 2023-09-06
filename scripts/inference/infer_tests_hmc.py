import argparse
import yaml
import time
from pathlib import Path
from sunbird.inference import HMC 

# Run this file with defaults
# Set up bash scripts to automatically send 4 variations per job

if __name__ == "__main__":
    output_path = Path('/n/holystore01/LABS/itc_lab/Users/ccuestalazaro/sunbird/chains/emulator_paper/')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, default="configs/hmc_2pcf.yaml"
    )
    
    # Make sure it reads from dataset with fixed hod
    parser.add_argument('--observation', type=str, default='Abacus')
    parser.add_argument('--cosmology', type=int, default=0,)
    parser.add_argument('--hod_idx', type=int, default=26,)
    parser.add_argument('--loss', type=str, default='mae')
    parser.add_argument("--s_min", type=float, default=0.7)
    parser.add_argument("--s_max", type=float, default=150.)
    parser.add_argument(
            "--multipoles",
            action="store",
            type=int,
            default=[0, 2],
            nargs="+",
        )
    parser.add_argument(
            "--quintiles",
            action="store",
            type=int,
            default=[0, 1, 3, 4],
            nargs="+",
        )
    parser.add_argument(
        "--statistics", 
            action="store",
            type=str,
            default=['density_split_cross', 'density_split_auto'],
            nargs="+",
    )
    parser.add_argument('--assembly_bias', action=argparse.BooleanOptionalAction, default=True,)
    parser.add_argument('--velocity_bias', action=argparse.BooleanOptionalAction, default=True,)
    parser.add_argument('--emulator_error', action=argparse.BooleanOptionalAction, default=True,)
    parser.add_argument('--predicted_uncertainty', action=argparse.BooleanOptionalAction, default=False,)
    parser.add_argument('--simulation_error', action=argparse.BooleanOptionalAction, default=True,)
    args = parser.parse_args()
    dir_name = (
        f"cos={args.cosmology}-h={args.hod_idx}-o={args.observation}-l={args.loss}-"
        f"smin={args.s_min}-smax={args.s_max}-m={''.join(map(str, args.multipoles))}-"
        f"q={''.join(map(str, args.quintiles))}-st={';'.join(args.statistics)}-"
        f"ab={int(args.assembly_bias or 0)}-vb={int(args.velocity_bias or 0)}-"
        f"ete={int(args.emulator_error or 0)}-se={int(args.simulation_error or 0)}"
    )
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    # Apply changes in config
    if args.observation == 'Uchuu':
        config['data']['observation']['class'] = 'Uchuu'
        config['data']['observation']['args'] = {} 
        config['data']['observation']['get_obs_args'] = {'ranking': 'random'}
    else:
        config['data']['observation']['get_obs_args']['cosmology'] = args.cosmology
        config['data']['observation']['get_obs_args']['hod_idx'] = args.hod_idx
    config['theory_model']['args']['loss'] = args.loss
    config['slice_filters']['s'] = [args.s_min, args.s_max]
    config['select_filters']['quintiles'] = args.quintiles
    config['select_filters']['multipoles'] = args.multipoles
    config['statistics'] = args.statistics

    if args.assembly_bias is False:
        config['fixed_parameters'] += ['B_cen', 'B_sat']
    if args.velocity_bias is False:
        config['fixed_parameters'] += ['alpha_s', 'alpha_c']
    if args.emulator_error: 
        config['data']['covariance']['add_emulator_error_test_set'] = True 
    else:
        config['data']['covariance']['add_emulator_error_test_set'] = False 
    if args.predicted_uncertainty: 
        config['data']['covariance']['add_predicted_uncertainty'] = True 
        dir_name += f'-predun={int(args.predicted_uncertainty) or 0)}'
    else:
        config['data']['covariance']['add_predicted_uncertainty'] = False 
    if args.simulation_error:
        config['data']['covariance']['add_simulation_error'] = True 
    else:
        config['data']['covariance']['add_simulation_error'] = False 

    # Store all args in folder name
    config["inference"]["output_dir"] = output_path / dir_name 
    print("output dir")
    print(config["inference"]["output_dir"])

    hmc = HMC.from_config_dict(
        config=config,
    )
    t0 = time.time()
    print(f"Fitting parameters {hmc.param_names}")
    hmc()
    print("Fitting took = ", time.time() - t0)
