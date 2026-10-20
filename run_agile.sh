#!/bin/bash
export PYTHONPATH="$HOME/agile_libs:$HOME/agile:$PYTHONPATH"
cd ~/agile
~/IsaacLab/isaaclab.sh -p "$@"
