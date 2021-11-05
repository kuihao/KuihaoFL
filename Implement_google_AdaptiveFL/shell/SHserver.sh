#!/bin/sh
# This file is called ~/script.sh
screen -S FL-server -dm bash -c 'python server.py -m 0;echo "**end of file**";$SHELL'