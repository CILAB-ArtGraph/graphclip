#!/bin/bash
sbatch -o ./outs/0_7/normal_clip_graph_resgated2.out -e ./outs/0_7/normal_clip_graph_resgated2.out -job-name 0_7_resgated ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/0_7/resgated/normal_clip_graph_resgated2.yaml" --graph
sbatch -o ./outs/0_7/normal_clip_graph_resgated3.out -e ./outs/0_7/normal_clip_graph_resgated3.out -job-name 0_7_resgated ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/0_7/resgated/normal_clip_graph_resgated3.yaml" --graph
sbatch -o ./outs/0_7/normal_clip_graph_resgated2_tanh.out -e ./outs/0_7/normal_clip_graph_resgated2_tanh.out -job-name 0_7_resgated ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/0_7/resgated/normal_clip_graph_resgated2_tanh.yaml" --graph
sbatch -o ./outs/0_7/normal_clip_graph_resgated3_tanh.out -e ./outs/0_7/normal_clip_graph_resgated3_tanh.out -job-name 0_7_resgated ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/0_7/resgated/normal_clip_graph_resgated3_tanh.yaml" --graph
sbatch -o ./outs/0_7/normal_clip_graph_sage3.out -e ./outs/0_7/normal_clip_graph_sage3.out -job-name 0_7_sage ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/0_7/sage/normal_clip_graph_sage3.yaml" --graph
sbatch -o ./outs/0_7/normal_clip_graph_sage2.out -e ./outs/0_7/normal_clip_graph_sage2.out -job-name 0_7_sage ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/0_7/sage/normal_clip_graph_sage2.yaml" --graph
sbatch -o ./outs/0_7/normal_clip_graph_transformer3.out -e ./outs/0_7/normal_clip_graph_transformer3.out -job-name 0_7_transformer ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/0_7/transformer/normal_clip_graph_transformer3.yaml" --graph
sbatch -o ./outs/0_7/normal_clip_graph_transformer2.out -e ./outs/0_7/normal_clip_graph_transformer2.out -job-name 0_7_transformer ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/0_7/transformer/normal_clip_graph_transformer2.yaml" --graph
sbatch -o ./outs/0_7/normal_clip_graph_gat2.out -e ./outs/0_7/normal_clip_graph_gat2.out -job-name 0_7_gat ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/0_7/gat/normal_clip_graph_gat2.yaml" --graph
sbatch -o ./outs/0_7/normal_clip_graph_gat3.out -e ./outs/0_7/normal_clip_graph_gat3.out -job-name 0_7_gat ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/0_7/gat/normal_clip_graph_gat3.yaml" --graph