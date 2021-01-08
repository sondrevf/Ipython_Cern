#!/bin/bash

source /cvmfs/sft.cern.ch/lcg/views/LCG_93python3/x86_64-centos7-gcc7-opt/setup.sh
export PYTHONPATH=$PYTHONPATH:/afs/cern.ch/work/x/xbuffat/BimBim_workspace/BimBim
python3 /afs/cern.ch/work/s/sfuruset/08_LHCDipoleEta/LHCDipoleMoment_HTC.py $1 $2 $3 $4 $5
