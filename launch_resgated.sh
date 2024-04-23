#!/bin/bash

sbatch  -o res3_sigmoid_experiment.out -e res3_sigmoid_experiment.out ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/resgated/normal_clip_graph_resgated3.yaml" --graph
sbatch  -o res3_tanh_experiment.out -e res3_tanh_experiment.out ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/resgated/normal_clip_graph_resgated3_tanh.yaml" --graph
sbatch  -o res2_sigmoid_experiment.out -e res2_sigmoid_experiment.out ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/resgated/normal_clip_graph_resgated2.yaml" --graph
sbatch  -o res2_tanh_experiment.out -e res2_tanh_experiment.out ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/resgated/normal_clip_graph_resgated3_tanh.yaml" --graph