#!/bin/bash
mkdir ./outs/genre
sbatch -o ./outs/genre/normal_clip_graph_sage3.out -e ./outs/genre/normal_clip_graph_sage3.out --job-name genre_sage ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/genre/normal_clip_graph_sage3.yaml" --graph
sbatch -o ./outs/genre/normal_clip_graph_sage2.out -e ./outs/genre/normal_clip_graph_sage2.out --job-name genre_sage ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/genre/normal_clip_graph_sage2.yaml" --graph
sbatch -o ./outs/genre/normal_clip_graph_gat2.out -e ./outs/genre/normal_clip_graph_gat2.out --job-name genre_gat ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/genre/normal_clip_graph_gat2.yaml" --graph
sbatch -o ./outs/genre/normal_clip_graph_gat3.out -e ./outs/genre/normal_clip_graph_gat3.out --job-name genre_gat ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/genre/normal_clip_graph_gat3.yaml" --graph