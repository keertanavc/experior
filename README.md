# Experts as Prior

Run the following code to generate the sweep id:

 `wandb sweep sweeps/test_sweep.yaml -p experior-default -e vdblm`

 which generates `$sweep_id`. Then, run the following code:

 `sbatch slurm/normal_launch.slrm  wandb agent --count 1 vdblm/experior-default/$sweep_id`

 Run all the commands from this directory.
