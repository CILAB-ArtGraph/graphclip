#!/bin/bash

sbatch -o outs/baseline_style_naive.out -e outs/baseline_style_naive.out ./slurms/baselines/no_training/launch --parameters="configs_cineca/baselines/style/normal_clip_no_training.yaml"
sbatch -o outs/baseline_style_wiki.out -e outs/baseline_style_wiki.out ./slurms/baselines/no_training/launch --parameters="configs_cineca/baselines/style/normal_clip_no_training_wikipedia.yaml"
sbatch -o outs/baseline_style_mistral.out -e outs/baseline_style_mistral.out ./slurms/baselines/no_training/launch --parameters="configs_cineca/baselines/style/normal_clip_no_training_mistral.yaml"