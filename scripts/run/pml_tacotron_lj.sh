# 0 preparation
NAME=tacotron-lj-pml #////////////////////////////////////TBC

# 0.1 set up environment
. /home/miproj/4thyr.oct2018/je369/.bashrc # we need to add conda commands to the path by running bashrc setup

# python environment
conda info -e # print list of current environments for debug
conda activate /home/miproj/4thyr.oct2018/je369/workspace/implementations/tacotron/venv/

# cuda environment
export PATH=${PATH}:/usr/local/cuda-9.0/bin
export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE

# 0.2 ld preload with tc malloc for performance
unset LD_PRELOAD
export LD_PRELOAD=/home/miproj/4thyr.oct2018/je369/workspace/implementations/tacotron/lib/gperftools/.libs/libtcmalloc.so

# 0.3 tools for dealing with PML features
export ProjectDir=/home/miproj/4thyr.oct2018/je369/tools/merlin_je369
export PYTHONPATH=${PYTHONPATH}:${ProjectDir}:${ProjectDir}/straight:${ProjectDir}/pulsemodel:${ProjectDir}/sigproc:${ProjectDir}/pulsemodel/external/REAPER/build
export PATH=${PATH}:${ProjectDir}/straight/analysis:${ProjectDir}/SPTK-3.7/bin

# 1 for running on grid: copy data to air
cd /home/miproj/4thyr.oct2018/je369/workspace/implementations/tacotron/scripts
python remove_training_data.py --train_dir pml-training
python check_move_data.py

# 2 preparation for the main code

# 2.1 for running on grid: go to right directory
cd /home/miproj/4thyr.oct2018/je369/workspace/implementations/tacotron

# 2.2 run the preprocess script
which python
python preprocess.py --base_dir /scratch/je369/tacotron/ --dataset ljspeech --output pml-training # ljspeech used as dataset key

# 2.3 remove the original data
cd /home/miproj/4thyr.oct2018/je369/workspace/implementations/tacotron/scripts
python remove_source_data.py

# 3 train the model

# 3.1 create the directory needed for logging
cd /home/miproj/4thyr.oct2018/je369/workspace/implementations/tacotron/scripts
python create_log_dir.py

# 3.2 run the train scheme
cd /home/miproj/4thyr.oct2018/je369/workspace/implementations/tacotron
python train.py --model tacotron_pml --base_dir /scratch/je369/tacotron --name $NAME --log_dir /scratch/je369/results --num_steps 500000 --input pml-training/train.txt --slack_url https://hooks.slack.com/services/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

# 3.3 move the logs and results back to the home directory
cd /home/miproj/4thyr.oct2018/je369/workspace/implementations/tacotron/scripts
python move_log_dir.py --name $NAME

# 4 for running on grid: remove data from air (optional)
cd /home/miproj/4thyr.oct2018/je369/workspace/implementations/tacotron/scripts
python remove_training_data.py --train_dir pml-training
