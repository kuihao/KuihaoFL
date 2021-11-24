#!/bin/sh
# This file is called ~/script.sh
#screen -S FL-server -m bash -c 'python kuihao_fl_sequential_multi_clients_simulation.py -m 0 -n cifar100_noniid_resnet18_fedavgm_10client_4000round_mountum09_leta1em3d2_seta1e0;echo "**end of file**";$SHELL'
# python kuihao_fl_sequential_multi_clients_simulation.py -m 0 -n cifar100_noniid_resnet18_fedyogi
screen -S FL-server -m bash -c 'python kuihao_fl_sequential_multi_clients_simulation.py -m 0 -n cifar100_noniid_resnet18_fedyogi;echo "**end of file**";$SHELL'