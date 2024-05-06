#!/bin/bash

sbatch -o outs/baseline_genre_naive.out -e outs/baseline_genre_naive.out ./slurms/baselines/no_training/launch --parameters="configs_cineca/baselines/genre/normal_clip_no_training.yaml"
sbatch -o outs/baseline_genre_wiki.out -e outs/baseline_genre_wiki.out ./slurms/baselines/no_training/launch --parameters="configs_cineca/baselines/genre/normal_clip_no_training_wikipedia.yaml"
sbatch -o outs/baseline_genre_mistral.out -e outs/baseline_genre_mistral.out ./slurms/baselines/no_training/launch --parameters="configs_cineca/baselines/genre/normal_clip_no_training_mistral.yaml"