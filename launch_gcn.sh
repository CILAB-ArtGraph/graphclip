#!/bin/bash

sbatch  -o gcn3experiment.out -e gcn3experiment.out ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/gcn/normal_clip_graph_gcn3.yaml" --graph
sbatch  -o gcn2experiment.out -e gcn2experiment.out ./slurms/proposed_model/launch --parameters="./configs_cineca/proposed_model/gcn/normal_clip_graph_gcn2.yaml" --graph