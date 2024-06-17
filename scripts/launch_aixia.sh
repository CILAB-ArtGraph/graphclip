#!/bin/bash
source .venv/bin/activate

echo "STYLE"
python main.py experiment --parameters configs_cineca/baselines/style/normal_aixia.yaml --aixia

echo "GENRE"
python main.py experiment --parameters configs_cineca/baselines/genre/normal_aixia.yaml --aixia