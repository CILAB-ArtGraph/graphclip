#!/bin/bash

sbatch  -o gat3experiment.out -e gat3experiment.out --job-name gat3exp ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/gat/normal_clip_graph_gat3.yaml" --graph
sbatch  -o gat2experiment.out -e gat2experiment.out --job-name gat3exp ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/gat/normal_clip_graph_gat2.yaml" --graph