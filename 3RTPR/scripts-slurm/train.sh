#!/bin/bash
#SBATCH --job-name=TIP.Rev1 # define job name
#SBATCH --nodes=1             # define node
#SBATCH --gpus-per-node=1     # define gpu limmit in 1 node
#SBATCH --ntasks=1            # define number tasks
#SBATCH --cpus-per-task=8    # There are 24 CPU cores
#SBATCH --time=7-23:00:00     # Max running time = 10 minutes
#SBATCH --output=xxxxxx.log
#SBATCH --nodelist=node004
# Load module
# Some module avail:
# source env/bin/activate

# ## pytorch-extra-py39-cuda11.2-gcc9
# module load cuda11.2/toolkit/11.2.2
# module load pytorch-py39-cuda11.2-gcc9/1.9.1
# module load pytorch-extra-py39-cuda11.2-gcc9
# module load opencv4-py39-cuda11.2-gcc9/4.5.4








#base model - top k
for k in 0 0.1 0.2 0.3 0.4 0.5; do
python run.py  --cfg config_model.yml \
  --d-names CUHK-PEDES -n hihi  --l-name sdm  --bs 128 --saug-text --erpt 0.2 \
  --lossweight-sdm 1 --sratio $k --fusedim 4096 


python run.py  --cfg config_model.yml \
  --d-names ICFG-PEDES -n hihi  --l-name sdm  --bs 128 --saug-text --erpt 0.2 \
  --lossweight-sdm 1 --sratio $k --fusedim 4096 


python run.py  --cfg config_model.yml \
  --d-names RSTPReid -n hihi  --l-name sdm  --bs 128 --saug-text --erpt 0.2 \
  --lossweight-sdm 1 --sratio $k --fusedim 4096
done


#base model - top z
for k in 512 1024 2048 8192; do
python run.py  --cfg config_model.yml \
  --d-names CUHK-PEDES -n hihi  --l-name sdm  --bs 128 --saug-text --erpt 0.2 \
  --lossweight-sdm 1 --sratio 0.3 --fusedim $k 


python run.py  --cfg config_model.yml \
  --d-names ICFG-PEDES -n hihi  --l-name sdm  --bs 128 --saug-text --erpt 0.2 \
  --lossweight-sdm 1 --sratio 0.3 --fusedim $k 


python run.py  --cfg config_model.yml \
  --d-names RSTPReid -n hihi  --l-name sdm  --bs 128 --saug-text --erpt 0.2 \
  --lossweight-sdm 1 --sratio 0.3 --fusedim $k
done


# #base + ALR
python run.py  --cfg config_model.yml \
  --d-names CUHK-PEDES -n hihi  --l-name sdm  --bs 128 --saug-text --erpt 0.2 \
  --ccd --ldynamic  --ldynamic-t -1 --ldynamic-m min --lossweight-sdm 1 --sratio 0.2 --fusedim 2048 


python run.py  --cfg config_model.yml \
  --d-names ICFG-PEDES -n hihi  --l-name sdm  --bs 128 --saug-text --erpt 0.2 \
  --ccd --ldynamic  --ldynamic-t -1 --ldynamic-m min --lossweight-sdm 1 --sratio 0.2 --fusedim 2048 


python run.py  --cfg config_model.yml \
  --d-names RSTPReid -n hihi  --l-name sdm  --bs 128 --saug-text --erpt 0.2 \
  --ccd --ldynamic  --ldynamic-t -1 --ldynamic-m min --lossweight-sdm 1 --sratio 0.2 --fusedim 2048 
