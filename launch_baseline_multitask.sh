#!/bin/bash

sbatch -o outs/baseline_multitask_naive_ft.out -e outs/baseline_multitask_naive_ft.out ./slurms/baselines/fine_tuning/launch --parameters="configs_cineca/baselines/multitask/normal_clip_fine_tuning_naive.yaml" --multitask
sbatch -o outs/baseline_multitask_wiki_ft.out -e outs/baseline_multitask_wiki_ft.out ./slurms/baselines/fine_tuning/launch --parameters="configs_cineca/baselines/multitask/normal_clip_fine_tuning_wikipedia.yaml" --multitask
sbatch -o outs/baseline_multitask_mistral_ft.out -e outs/baseline_multitask_mistral_ft.out ./slurms/baselines/fine_tuning/launch --parameters="configs_cineca/baselines/multitask/normal_clip_fine_tuning_mistral.yaml" --multitask
