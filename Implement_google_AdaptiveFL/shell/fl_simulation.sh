#!/bin/sh
# This file is called ~/script.sh
screen -S FL-server -m bash -c 'python kuihao_fl_sequential_multi_clients_simulation.py -m 0 -n cifar100_noniid_resnet18_fedavg_10client_4000round_leta1em1_seta1e1;echo "**end of file**";$SHELL'