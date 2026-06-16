#!/bin/bash -l
#SBATCH --partition=course_gpu
#SBATCH --account=2026-spring-ds-677-amr239-sd2448
#SBATCH --qos=course
#SBATCH --job-name=emomusicgen
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/course/2026/spring/ds/677/amr239/sd2448/emomusicgen/logs/train_%j.out
#SBATCH --error=/course/2026/spring/ds/677/amr239/sd2448/emomusicgen/logs/train_%j.err

module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /home/sd2448/.conda/envs/emomusicvae

cd /course/2026/spring/ds/677/amr239/sd2448/emomusicgen
python train.py
