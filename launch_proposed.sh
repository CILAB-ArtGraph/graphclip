#!/bin/bash

rm -rf models/proposed/*
rm -rf outs/*/*.out

./scripts/launch_0_1.sh
./scripts/launch_0_3.sh
./scripts/launch_0_5.sh
./scripts/launch_0_7.sh
./scripts/launch_1_0.sh