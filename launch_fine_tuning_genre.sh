#!/bin/bash

sbatch -o outs/baseline_genre_naive_ft.out -e outs/baseline_genre_naive_ft.out ./slurms/baselines/fine_tuning/launch --parameters="configs_cineca/baselines/genre/normal_clip_fine_tuning_naive.yaml"
sbatch -o outs/baseline_genre_wiki_ft.out -e outs/baseline_genre_wiki_ft.out ./slurms/baselines/fine_tuning/launch --parameters="configs_cineca/baselines/genre/normal_clip_fine_tuning_wikipedia.yaml"
sbatch -o outs/baseline_genre_mistral_ft.out -e outs/baseline_genre_mistral_ft.out ./slurms/baselines/fine_tuning/launch --parameters="configs_cineca/baselines/genre/normal_clip_fine_tuning_mistral.yaml"
