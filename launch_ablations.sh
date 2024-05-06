#!/bin/bash

mkdir outs/ablations
sbatch -o ./outs/ablations/vit_style.out -e ./outs/ablations/vit_style.out --job-name vit_style ./slurms/baselines/fine_tuning/launch --parameters="./configs_cineca/baselines/style/vit.yaml" --ablation
sbatch -o ./outs/ablations/vit_genre.out -e ./outs/ablations/vit_genre.out --job-name vit_genre ./slurms/baselines/fine_tuning/launch --parameters="./configs_cineca/baselines/genre/vit.yaml" --ablation