#!/bin/sh
# This file is called ~/script.sh
#pkill screen
screen -S client-0 -dm bash -c 'python client.py -c 0 --gpu 0;echo "**end of file1**";$SHELL'
sleep 2s
screen -S client-1 -dm bash -c 'python client.py -c 1 --gpu 0;echo "**end of file2**";$SHELL'
sleep 2s
screen -S client-2 -dm bash -c 'python client.py -c 2 --gpu 0;echo "**end of file3**";$SHELL'
sleep 2s
screen -S client-3 -dm bash -c 'python client.py -c 3 --cpu;echo "**end of file4**";$SHELL'
sleep 2s
screen -S client-4 -dm bash -c 'python client.py -c 4 --cpu;echo "**end of file5**";$SHELL'
sleep 2s
screen -S client-5 -dm bash -c 'python client.py -c 5 --gpu 1;echo "**end of file6**";$SHELL'
sleep 2s
screen -S client-6 -dm bash -c 'python client.py -c 6 --gpu 1;echo "**end of file7**";$SHELL'
sleep 2s
screen -S client-7 -dm bash -c 'python client.py -c 7 --gpu 1;echo "**end of file8**";$SHELL'
sleep 2s
screen -S client-8 -dm bash -c 'python client.py -c 8 --cpu;echo "**end of file9**";$SHELL'
sleep 2s
screen -S client-9 -dm bash -c 'python client.py -c 9 --cpu;echo "**end of file10**";$SHELL'