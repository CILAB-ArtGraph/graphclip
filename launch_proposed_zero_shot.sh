#!/bin/bash
mkdir ./outs/style
sbatch -o ./outs/style/zero_shot_clip_graph_sage3.out -e ./outs/style/zero_shot_clip_graph_sage3.out --job-name zss3 ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/style/zero_shot_clip_graph_sage3.yaml" --graph
sbatch -o ./outs/zero_shot/zero_shot_clip_graph_sage2.out -e ./outs/style/zero_shot_clip_graph_sage2.out --job-name zss2 ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/style/zero_shot_clip_graph_sage2.yaml" --graph
sbatch -o ./outs/zero_shot/zero_shot_clip_graph_gat2.out -e ./outs/style/zero_shot_clip_graph_gat2.out --job-name zsg2 ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/style/zero_shot_clip_graph_gat2.yaml" --graph
sbatch -o ./outs/zero_shot/zero_shot_clip_graph_gat3.out -e ./outs/style/zero_shot_clip_graph_gat3.out --job-name zsg3 ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/style/zero_shot_clip_graph_gat3.yaml" --graph

mkdir ./outs/genre
sbatch -o ./outs/genre/zero_shot_clip_graph_sage3.out -e ./outs/genre/zero_shot_clip_graph_sage3.out --job-name zgs3 ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/genre/zero_shot_clip_graph_sage3.yaml" --graph
sbatch -o ./outs/genre/zero_shot_clip_graph_sage2.out -e ./outs/genre/zero_shot_clip_graph_sage2.out --job-name zgs2 ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/genre/zero_shot_clip_graph_sage2.yaml" --graph
sbatch -o ./outs/genre/zero_shot_clip_graph_gat2.out -e ./outs/genre/zero_shot_clip_graph_gat2.out --job-name zgg2 ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/genre/zero_shot_clip_graph_gat2.yaml" --graph
sbatch -o ./outs/genre/zero_shot_clip_graph_gat3.out -e ./outs/genre/zero_shot_clip_graph_gat3.out --job-name zgg3 ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/genre/zero_shot_clip_graph_gat3.yaml" --graph