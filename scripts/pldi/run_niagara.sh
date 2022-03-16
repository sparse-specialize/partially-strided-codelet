#!/bin/sh


#SBATCH --cpus-per-task=40
#SBATCH --export=ALL
#SBATCH --job-name="TRSV"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=kazem.cheshmi@gmail.com
#SBATCH --nodes=1
#SBATCH --output="DDT.%j.%N.out"
#SBATCH -t 12:00:00

#### NOTE #####
# ######################################
# ######################################
# ######################################
# ######################################
# ######################################
# ######################################
#
# THIS SCRIPT ASSUMES THE BINARY FILES ARE COMPILED
#
# ######################################
# ######################################
# ######################################
# ######################################
# ######################################
# ######################################
# ######################################

if command -v module; then
  module load NiaEnv/2019b
  module load cmake/3.17.3
  module load intel
  module load intel/2019u4
  module load intel
  module load gcc/9.4.0 
 module load metis/5.1.0
  module load mkl
fi

#export MKLROOT=/scinet/intel/2019u4/compilers_and_libraries_2019.4.243/linux/mkl
#export SUITEROOT=/home/m/mmehride/kazem/programs/SuiteSparse/
#export METISROOT=/scinet/niagara/software/2019b/opt/intel-2019u4/metis/5.1.0/

##### THESE SHOULD BE CHANGED FOR NIAGARA
BINPATH=/scratch/m/mmehride/kazem/development/codelet_mining/build/demo # /home/cetinicz/CLionProjects/DDT/cmake-build-release/demo
LOGS=/scratch/m/mmehride/kazem/development/codelet_mining/build/demo/logs # /home/cetinicz/CLionProjects/DDT/scripts
SCRIPTPATH=/scratch/m/mmehride/kazem/development/codelet_mining/scripts/pldi # /home/cetinicz/CLionProjects/DDT/scripts
MAT_DIR=/scratch/m/mmehride/kazem/development/codelet_mining/scripts/ssgetpy/mm
SPD_MAT_DIR=/media/HDD/matrix

CURRENT_TIME=$(date +%s)

# bash $SCRIPTPATH/run_exp.sh $BINPATH/spmv_demo   3 20  $MAT_DIR $SPD_MAT_DIR #> $LOGS/spmv_$CURRENT_TIME.csv
#bash $SCRIPTPATH/run_exp.sh $BINPATH/sptrsv_demo 1 20 $MAT_DIR $SPD_MAT_DIR > $LOGS/sptrsv_$CURRENT_TIME.csv
bash $SCRIPTPATH/run_exp.sh $BINPATH/spmm_demo   4 20  $MAT_DIR $SPD_MAT_DIR  > $LOGS/spmm_$CURRENT_TIME.csv
