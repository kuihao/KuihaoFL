#!/bin/sh
# This file is called ~/script.sh
screen -S FL-server -m bash -c 'python server.py -m 0 -n ec_resnet18_fedadagrad_10client_1000round_GRAM2048_eta1e3_tau1e2 --cpu;echo "**end of file**";$SHELL'