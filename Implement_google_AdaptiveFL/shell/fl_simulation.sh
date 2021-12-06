#!/bin/sh
# This file is called ~/script.sh                                
screen -S FL-server -m bash -c 'python centralize_learning_fl_simulation.py -m 0 -n 600round_centro_cifar100_noniid_resnet18_SGD;echo "**end of file**";$SHELL'
#screen -S FL-server -m bash -c 'python kuihao_fl_sequential_multi_clients_simulation.py -m 0 -n 4000round_cifar100_noniid_resnet18_fedadagrad;echo "**end of file**";$SHELL'

# load checkpoint
# python server.py -m 2 -cp CheckpointPath -n modelname
#screen -S FL-server -m bash -c 'python centralize_learning_fl_simulation.py -m 2 -cp ckpoint/12_05_2021__17_13_33_NewCode_400round_centro_cifar100_noniid_resnet18_SGDM -n continue_400add400round_centro_cifar100_noniid_resnet18_SGDM;echo "**end of file**";$SHELL'

# Recover from interrupt
# python server.py -m 1 -pmn PriorModelName -n modelname
#python centralize_learning_fl_simulation.py -m 1 -pmn 12_06_2021__13_11_45_TestRecovery_400round_centro_cifar100_noniid_resnet18_SGDM -n Recovery_400round_centro_cifar100_noniid_resnet18_SGDM
##screen -S FL-server -m bash -c ';echo "**end of file**";$SHELL'