#!/bin/bash

sbatch  -o transformer3experiment.out -e transformer3experiment.out --job-name trans3exp ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/transformer/normal_clip_graph_transformer3.yaml" --graph
sbatch  -o transformer2experiment.out -e transformer2experiment.out --job-name trans2exp ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/transformer/normal_clip_graph_transformer2.yaml" --graph