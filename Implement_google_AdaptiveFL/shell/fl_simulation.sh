#!/bin/sh
# This file is called ~/script.sh                                
#screen -S FL-server -m bash -c 'python centralize_learning_fl_simulation.py -m 0 -n 400round_centro_cifar100_noniid_resnet18_fedyogi;echo "**end of file**";$SHELL'
screen -S FL-server -m bash -c 'python kuihao_fl_sequential_multi_clients_simulation.py -m 0 -n 4000round_cifar100_noniid_resnet18_fedadagrad;echo "**end of file**";$SHELL'