#!/bin/bash

loss_value="mae"

commands=(
    "python infer_tests_hmc.py --statistics tpcf density_split_cross density_split_auto --loss $loss_value"
    "python infer_tests_hmc.py --statistics tpcf density_split_cross density_split_auto --s_max 30. --loss $loss_value"
    "python infer_tests_hmc.py --statistics tpcf density_split_cross density_split_auto --s_min 30. --loss $loss_value"
    "python infer_tests_hmc.py --statistics tpcf density_split_cross density_split_auto --quintiles 0  --loss $loss_value"
    "python infer_tests_hmc.py --statistics tpcf density_split_cross density_split_auto --quintiles 4  --loss $loss_value"
    "python infer_tests_hmc.py --statistics tpcf density_split_cross density_split_auto --quintiles 0 4 --loss $loss_value"
    "python infer_tests_hmc.py --statistics tpcf density_split_cross density_split_auto --multipoles 0 --loss $loss_value"
    "python infer_tests_hmc.py --statistics tpcf density_split_cross density_split_auto --multipoles 2  --loss $loss_value"
    "python infer_tests_hmc.py --statistics tpcf density_split_cross density_split_auto --no-assembly_bias --loss $loss_value"
    "python infer_tests_hmc.py --statistics tpcf density_split_cross density_split_auto --no-velocity_bias --loss $loss_value"
    "python infer_tests_hmc.py --statistics tpcf density_split_cross density_split_auto --no-emulator_error --loss $loss_value"
    "python infer_tests_hmc.py --statistics tpcf density_split_cross density_split_auto --no-simulation_error --loss $loss_value"
    "python infer_tests_hmc.py --statistics tpcf density_split_cross density_split_auto --no-emulator_error --no-simulation_error --loss $loss_value"

    "python infer_tests_hmc.py --statistics tpcf --observation Uchuu --loss $loss_value"
    "python infer_tests_hmc.py --statistics tpcf density_split_auto density_split_cross --observation Uchuu --loss $loss_value"

    "python infer_tests_hmc.py --statistics tpcf --loss $loss_value"
    "python infer_tests_hmc.py --statistics tpcf density_split_cross --loss $loss_value"
    "python infer_tests_hmc.py --statistics density_split_auto --loss $loss_value"
    "python infer_tests_hmc.py --statistics density_split_cross --loss $loss_value"
    "python infer_tests_hmc.py --statistics density_split_cross density_split_auto --loss $loss_value"

    "python infer_tests_hmc.py --statistics tpcf density_split_cross density_split_auto --cosmology 1 --hod_idx 74 --loss $loss_value"
    "python infer_tests_hmc.py --statistics tpcf density_split_cross density_split_auto --cosmology 3 --hod_idx 30 --loss $loss_value"
    "python infer_tests_hmc.py --statistics tpcf density_split_cross density_split_auto --cosmology 4 --hod_idx 15 --loss $loss_value"
)

for (( i=0; i<${#commands[@]}; i+=4 )); do
    batch_file="batch_$((i / 4)).sh"
    
    echo "#!/bin/bash" > $batch_file
    echo "#SBATCH --job-name=hmc_test_$((i / 4))" >> $batch_file
    echo "#SBATCH --output=logs/slurm-%j.out" >> $batch_file
    echo "#SBATCH --nodes=1" >> $batch_file
    echo "#SBATCH --ntasks=1" >> $batch_file
    echo "#SBATCH --time=01:00:00" >> $batch_file
    echo "#SBATCH --partition=shared" >> $batch_file
    echo "#SBATCH --mem-per-cpu=1G" >> $batch_file

    for (( j=0; j<4 && i+j<${#commands[@]}; j++ )); do
        echo "${commands[i+j]}" >> $batch_file
    done
    
    sbatch $batch_file
done

