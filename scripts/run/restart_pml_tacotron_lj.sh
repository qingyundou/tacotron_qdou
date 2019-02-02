# 0 preparation
NAME=tacotron-lj-pml #////////////////////////////////////TBC

# 0.1 set up environment
. /home/miproj/4thyr.oct2018/je369/.bashrc # we need to add conda commands to the path by running bashrc setup
conda info -e # print list of current environments for debug
conda activate /home/miproj/4thyr.oct2018/je369/workspace/implementations/tacotron/venv/
export PATH=${PATH}:/usr/local/cuda-9.0/bin
unset LD_PRELOAD
export LD_PRELOAD=/home/miproj/4thyr.oct2018/je369/workspace/implementations/tacotron/lib/gperftools/.libs/libtcmalloc.so
export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE

# 1 restart training the model

# 1.1 run the train scheme
cd /home/miproj/4thyr.oct2018/je369/workspace/implementations/tacotron
python train_pml.py --model tacotron_pml --base_dir /scratch/je369/tacotron --name $NAME --log_dir /scratch/je369/results --num_steps 500000 --restore_step 270000

# 1.2 move the logs and results back to the home directory
cd /home/miproj/4thyr.oct2018/je369/workspace/implementations/tacotron/scripts
python move_log_dir.py --name $NAME

# 2 for running on grid: remove data from air (optional)
cd /home/miproj/4thyr.oct2018/je369/workspace/implementations/tacotron/scripts
python remove_training_data.py
