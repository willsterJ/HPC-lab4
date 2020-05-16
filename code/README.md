# HPC lab 2
Code runs on

python 3.6.3

pytorch 0.3.0_4

To run on HPC Prince: 
srun --reservation=cds-courses --gres=gpu:k80:4 --nodes=1 --cpus-per-task=28 --time=04:00:00 --mem=250GB --pty /bin/bash

python3 ./lab4 --batch_size 128 --gpu_count 4
