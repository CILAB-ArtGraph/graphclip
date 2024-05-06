#!/bin/bash
mkdir ./outs/style
sbatch -o ./outs/style/normal_clip_graph_sage3.out -e ./outs/style/normal_clip_graph_sage3.out --job-name style_sage ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/style/normal_clip_graph_sage3.yaml" --graph
sbatch -o ./outs/style/normal_clip_graph_sage2.out -e ./outs/style/normal_clip_graph_sage2.out --job-name style_sage ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/style/normal_clip_graph_sage2.yaml" --graph
sbatch -o ./outs/style/normal_clip_graph_gat2.out -e ./outs/style/normal_clip_graph_gat2.out --job-name style_gat ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/style/normal_clip_graph_gat2.yaml" --graph
sbatch -o ./outs/style/normal_clip_graph_gat3.out -e ./outs/style/normal_clip_graph_gat3.out --job-name style_gat ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/style/normal_clip_graph_gat3.yaml" --graph