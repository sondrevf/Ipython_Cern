loc_output_path="/home/sfuruset/hdd/01_Python/02_IPython/01_DistributionDiffusion/01_Output/"
afs_output_path="/afs/cern.ch/work/s/sfuruset/07_PyRADISE/01_Output/"


rsync -Pu sfuruset@lxplus.cern.ch:$afs_output_path*_S2* $loc_output_path
