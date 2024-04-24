#!/bin/bash

sbatch -o outs/split_0_1.out -e outs/split_0_1.out --job-name split_0_1 ./slurms/preprocess/graph/split_graph/launch --parameters configs_cineca/preprocess/graph/split_graph_0_1.yaml --graph
sbatch -o outs/split_0_3.out -e outs/split_0_3.out --job-name split_0_3 ./slurms/preprocess/graph/split_graph/launch --parameters configs_cineca/preprocess/graph/split_graph_0_3.yaml --graph
sbatch -o outs/split_0_5.out -e outs/split_0_5.out --job-name split_0_5 ./slurms/preprocess/graph/split_graph/launch --parameters configs_cineca/preprocess/graph/split_graph_0_5.yaml --graph
sbatch -o outs/split_0_7.out -e outs/split_0_7.out --job-name split_0_7 ./slurms/preprocess/graph/split_graph/launch --parameters configs_cineca/preprocess/graph/split_graph_0_7.yaml --graph
sbatch -o outs/split_1_0.out -e outs/split_1_0.out --job-name split_1_0 ./slurms/preprocess/graph/split_graph/launch --parameters configs_cineca/preprocess/graph/split_graph_1_0.yaml --graph