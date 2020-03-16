#!/bin/bash

pwd=$apwd
eos_output_path="/eos/user/s/sfuruset/01_Synch_Work_Desktop/01_Python/02_IPython/01_DistributionDiffusion/01_Output/"
loc_output_path="/home/sfuruset/hdd/01_Python/02_IPython/01_DistributionDiffusion/01_Output/"
afs_output_path="/afs/cern.ch/work/s/sfuruset/07_PyRADISE/01_Output/"
hpc_output_path="/hpcscratch/user/sfuruset/07_PyRADISE/01_Output/"

#scp $eos_output_path$1 $afs_output_path

# Can put names inside the file or as input
filenames="
calc_Dt_ax2.55e-05_bx:ax-0.70_gx2e-02_BC0_2D_Kx1.0e-04_Ky0.0e+00_ibs0e+00-0e+00_iC3_reDQx-8.4e-05_imDQx4.3e-06_finda110_Nc700r_tmaxX_S0.pkl
calc_Dt_ax2.67e-05_bx:ax-0.70_gx2e-02_BC0_2D_Kx1.0e-04_Ky0.0e+00_ibs0e+00-0e+00_iC3_reDQx-8.4e-05_imDQx4.3e-06_finda110_Nc700r_tmaxX_S0.pkl
calc_Dt_ax2.92e-05_bx:ax-0.70_gx2e-02_BC0_2D_Kx1.0e-04_Ky0.0e+00_ibs0e+00-0e+00_iC3_reDQx-8.4e-05_imDQx4.3e-06_finda110_Nc700r_tmaxX_S0.pkl
calc_Dt_ax3.16e-05_bx:ax-0.70_gx2e-02_BC0_2D_Kx1.0e-04_Ky0.0e+00_ibs0e+00-0e+00_iC3_reDQx-8.4e-05_imDQx4.3e-06_finda110_Nc700r_tmaxX_S0.pkl
calc_Dt_ax3.40e-05_bx:ax-0.70_gx2e-02_BC0_2D_Kx1.0e-04_Ky0.0e+00_ibs0e+00-0e+00_iC3_reDQx-8.4e-05_imDQx4.3e-06_finda110_Nc700r_tmaxX_S0.pkl
calc_Dt_ax3.65e-05_bx:ax-0.70_gx2e-02_BC0_2D_Kx1.0e-04_Ky0.0e+00_ibs0e+00-0e+00_iC3_reDQx-8.4e-05_imDQx4.3e-06_finda110_Nc700r_tmaxX_S0.pkl
calc_Dt_ax3.89e-05_bx:ax-0.70_gx2e-02_BC0_2D_Kx1.0e-04_Ky0.0e+00_ibs0e+00-0e+00_iC3_reDQx-8.4e-05_imDQx4.3e-06_finda110_Nc700r_tmaxX_S0.pkl
calc_Dt_ax4.13e-05_bx:ax-0.70_gx2e-02_BC0_2D_Kx1.0e-04_Ky0.0e+00_ibs0e+00-0e+00_iC3_reDQx-8.4e-05_imDQx4.3e-06_finda110_Nc700r_tmaxX_S0.pkl
calc_Dt_ax4.37e-05_bx:ax-0.70_gx2e-02_BC0_2D_Kx1.0e-04_Ky0.0e+00_ibs0e+00-0e+00_iC3_reDQx-8.4e-05_imDQx4.3e-06_finda110_Nc700r_tmaxX_S0.pkl
calc_Dt_ax4.62e-05_bx:ax-0.70_gx2e-02_BC0_2D_Kx1.0e-04_Ky0.0e+00_ibs0e+00-0e+00_iC3_reDQx-8.4e-05_imDQx4.3e-06_finda110_Nc700r_tmaxX_S0.pkl
calc_Dt_ax4.86e-05_bx:ax-0.70_gx2e-02_BC0_2D_Kx1.0e-04_Ky0.0e+00_ibs0e+00-0e+00_iC3_reDQx-8.4e-05_imDQx4.3e-06_finda110_Nc700r_tmaxX_S0.pkl
"

filenames2=""
for name in $filenames
do 
    clean="${name//[$'\t\r\n']}"
    filenames2="$filenames2 ./$clean"
done
for var in "$@"
do
    filenames2="$filenames2 ./$var"
done

#filenames="${filenames//[$'\t\r\n']} ${@//[$'\t\r\n']}"


# Go to the output path
cd $loc_output_path

# Copy the files 
echo "scp $filenames2 sfuruset@lxplus.cern.ch:$afs_output_path"
scp $filenames2 sfuruset@lxplus.cern.ch:$afs_output_path

cd $pwd




