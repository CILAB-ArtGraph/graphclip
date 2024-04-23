#!/bin/bash

sbatch  -o res3_sigmoid_experiment.out -e res3_sigmoid_experiment.out --job-name ressig3exp ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/resgated/normal_clip_graph_resgated3.yaml" --graph
sbatch  -o res3_tanh_experiment.out -e res3_tanh_experiment.out --job-name restan3exp ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/resgated/normal_clip_graph_resgated3_tanh.yaml" --graph
sbatch  -o res2_sigmoid_experiment.out -e res2_sigmoid_experiment.out --job-name ressig2exp ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/resgated/normal_clip_graph_resgated2.yaml" --graph
sbatch  -o res2_tanh_experiment.out -e res2_tanh_experiment.out --job-name restan2exp ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/resgated/normal_clip_graph_resgated3_tanh.yaml" --graph