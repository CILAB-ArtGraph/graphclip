#!/bin/bash

rm -rf models/*
rm -rf outs/*/*.out

./launch_0_1.sh
./launch_0_3.sh
./launch_0_5.sh
./launch_0_7.sh
./launch_1_0.sh