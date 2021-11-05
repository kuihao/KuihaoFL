#!/bin/sh
# This file is called ~/script.sh
screen -S FL-server -m bash -c 'python server.py -m 0 --cpu;echo "**end of file**";$SHELL'