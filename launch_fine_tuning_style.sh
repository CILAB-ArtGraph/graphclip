#!/bin/bash

sbatch -o outs/baseline_style_naive_ft.out -e outs/baseline_style_naive_ft.out ./slurms/baselines/baselines/fine_tuning/launch --parameters="configs_cineca/baselines/style/normal_clip_fine_tuning_naive.yaml"
sbatch -o outs/baseline_style_wiki_ft.out -e outs/baseline_style_wiki_ft.out ./slurms/baselines/baselines/fine_tuning/launch --parameters="configs_cineca/baselines/style/normal_clip_fine_tuning_wikipedia.yaml"
sbatch -o outs/baseline_style_mistral_ft.out -e outs/baseline_style_mistral_ft.out ./slurms/baselines/baselines/fine_tuning/launch --parameters="configs_cineca/baselines/style/normal_clip_fine_tuning_mistral.yaml"
sbatch -o outs/baseline_style_prot_ft.out -e outs/baseline_style_prot_ft.out ./slurms/baselines/baselines/fine_tuning/launch --parameters="configs_cineca/baselines/style/normal_clip_fine_tuning_mistral_prototype.yaml"
