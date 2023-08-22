#!/bin/bash

loss_value="learned_gaussian"
cosmologies=(0 1 2 3 4 13)

commands=()

# Generate the commands
for cosmology in ${cosmologies[@]}; do
    for hod_idx in {0..99}; do
        commands+=("python infer_tests_hmc.py --statistics tpcf density_split_cross density_split_auto --cosmology $cosmology --hod_idx $hod_idx --loss $loss_value")
    done
done

# Create and submit batch files
for (( i=0; i<${#commands[@]}; i+=60 )); do
    batch_file="batch_$((i / 60)).sh"
    
    echo "#!/bin/bash" > $batch_file
    echo "#SBATCH --job-name=coverage_hmc_$((i / 60))" >> $batch_file
    echo "#SBATCH --output=logs/slurm-%j.out" >> $batch_file
    echo "#SBATCH --nodes=1" >> $batch_file
    echo "#SBATCH --ntasks=1" >> $batch_file
    echo "#SBATCH --time=06:00:00" >> $batch_file # 5-hour time limit
    echo "#SBATCH --partition=shared" >> $batch_file
    echo "#SBATCH --mem-per-cpu=1G" >> $batch_file

    for (( j=0; j<60 && i+j<${#commands[@]}; j++ )); do
        echo "${commands[i+j]}" >> $batch_file
    done
    
    sbatch $batch_file
done
