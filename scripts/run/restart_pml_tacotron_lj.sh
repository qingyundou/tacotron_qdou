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
export ProjectDir=/home/miproj/4thyr.oct2018/je369/workspace/implementations/tacotron
export PYTHONPATH=${PYTHONPATH}:${ProjectDir}:${ProjectDir}/straight:${ProjectDir}/pulsemodel:${ProjectDir}/sigproc:${ProjectDir}/pulsemodel/external/REAPER/build
export PATH=${PATH}:${ProjectDir}/straight/analysis:${ProjectDir}/SPTK-3.7/bin

# 1 restart training the model

# 1.1 run the train scheme
cd /home/miproj/4thyr.oct2018/je369/workspace/implementations/tacotro
python train.py --model tacotron_pml --base_dir /scratch/je369/tacotron --name $NAME --log_dir /scratch/je369/results --num_steps 500000 --restore_step 69000 --input gran-lj-training/train.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5" --slack_url https://hooks.slack.com/services/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

# 1.2 move the logs and results back to the home directory
cd /home/miproj/4thyr.oct2018/je369/workspace/implementations/tacotron/scripts
python move_log_dir.py --name $NAME

# 2 for running on grid: remove data from air (optional)
cd /home/miproj/4thyr.oct2018/je369/workspace/implementations/tacotron/scripts
# python remove_training_data.py --train_dir gran-lj-training
