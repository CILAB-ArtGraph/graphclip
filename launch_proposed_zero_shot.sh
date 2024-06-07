#!/bin/bash
mkdir ./outs/zero_shot
sbatch -o ./outs/zero_shot/normal_clip_graph_sage3.out -e ./outs/zero_shot/normal_clip_graph_sage3.out --job-name zss3 ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/style/zero_shot_clip_graph_sage3.yaml" --graph
sbatch -o ./outs/zero_shot/normal_clip_graph_sage2.out -e ./outs/zero_shot/normal_clip_graph_sage2.out --job-name zss2 ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/style/zero_shot_clip_graph_sage2.yaml" --graph
sbatch -o ./outs/zero_shot/normal_clip_graph_gat2.out -e ./outs/zero_shot/normal_clip_graph_gat2.out --job-name zsg2 ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/style/zero_shot_clip_graph_gat2.yaml" --graph
sbatch -o ./outs/zero_shot/normal_clip_graph_gat3.out -e ./outs/zero_shot/normal_clip_graph_gat3.out --job-name zsg3 ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/style/zero_shot_clip_graph_gat3.yaml" --graph