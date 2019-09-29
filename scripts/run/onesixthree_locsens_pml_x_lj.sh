# 0 preparation
NAME=tacotron-pml-x-163 #////////////////////////////////////TBC

# 0.1 set up environment
# . /home/miproj/4thyr.oct2018/je369/.bashrc # we need to add conda commands to the path by running bashrc setup
. /home/miproj/4thyr.oct2018/je369/miniconda3/etc/profile.d/conda.sh # I think only this line is needed

# python environment
conda info -e # print list of current environments for debug
conda activate /home/miproj/4thyr.oct2018/je369/workspace/implementations/tacotron/venv/

# cuda environment
export PATH=${PATH}:/usr/local/cuda-9.0/bin
export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE

# 0.2 ld preload with tc malloc for performance
unset LD_PRELOAD
export LD_PRELOAD=/home/miproj/4thyr.oct2018/je369/workspace/implementations/tacotron/lib/gperftools/.libs/libtcmalloc.so
# /home/dawna/tts/qd212/models/tacotron-master/lib/gperftools/.libs/libtcmalloc.so

# 0.3 tools for dealing with PML features
# /home/miproj/4thyr.oct2018/je369/workspace/implementations/tacotron
export ProjectDir=/home/dawna/tts/qd212/models/tacotron/
export PYTHONPATH=${PYTHONPATH}:${ProjectDir}:${ProjectDir}/lib:${ProjectDir}/lib/straight:${ProjectDir}/lib/pulsemodel:${ProjectDir}/lib/sigproc:${ProjectDir}/lib/pulsemodel/external/REAPER/build
export PATH=${PATH}:${ProjectDir}/lib/straight/analysis:${ProjectDir}/lib/SPTK-3.7/bin

# # 1 for running on grid: copy data to air
# cd ${ProjectDir}/scripts/
# python remove_training_data.py --train_dir 163-lj-training
# python check_move_data.py

# # 2 preparation for the main code

# # 2.1 for running on grid: go to right directory
# cd ${ProjectDir}

# # 2.2 run the preprocess script
# which python
# python preprocess.py --base_dir /scratch/je369/tacotron/ --dataset ljspeech --output 163-lj-training --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --validation_size 150 --test_size 150 # ljspeech used as dataset key

# # 2.3 remove the original data
# cd ${ProjectDir}/scripts/
# python remove_source_data.py

# # 3 train the model

# 3.1 create the directory needed for logging
cd ${ProjectDir}/scripts/
python create_log_dir.py

# 3.2 run the train scheme
cd ${ProjectDir}
URL=https://hooks.slack.com/services/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


############################ train with gta
# python train.py --variant tacotron_pml_x_locsens --base_dir /scratch/je369/tacotron --name $NAME --log_dir /scratch/qd212/tacotron/results --num_steps 150000 --tacotron_input 163-lj-training/train.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --slack_url https://hooks.slack.com/services/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

# python train.py --variant tacotron_pml_x_locsens --base_dir /scratch/je369/tacotron --name $NAME --log_dir /scratch/qd212/tacotron/results --num_steps 150000 --tacotron_input 163-lj-training/train.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --slack_url https://hooks.slack.com/services/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 --restore_step 98000

# python train.py --variant tacotron_pml_x_locsens --base_dir /scratch/qd212/tacotron --name tacotron-pml-x-163-merlin --log_dir ${ProjectDir}results --num_steps 150000 --tacotron_input 163-lj-training/train_merlin.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --slack_url ${URL} --restore_step 104000


# python train.py --variant tacotron_bk2orig --base_dir /scratch/je369/tacotron --name tacotron-bk2orig --log_dir ${ProjectDir}results --num_steps 150000 --tacotron_input 163-lj-training/train.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --slack_url ${URL} --restore_step 111000



############################ train with eal
# python train.py --variant tacotron_pml_x_locsens --base_dir /scratch/je369/tacotron --name tacotron-pml-x-163-eal-init --log_dir /scratch/qd212/tacotron/results --num_steps 150000 --tacotron_input 163-lj-training/train.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --slack_url ${URL} --eal_dir ${ProjectDir}results/tacotron-pml-x-163/gta/alignment/npy/ --eal_ckpt /home/miproj/4thyr.oct2018/je369/results/tacotron-schedsamp-pml-x-163/model.ckpt-150000

# python train.py --variant tacotron_pml_x_locsens --base_dir /scratch/je369/tacotron --name tacotron-pml-x-163-eal-initFT --log_dir ${ProjectDir}results --num_steps 300000 --tacotron_input 163-lj-training/train.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --slack_url ${URL} --eal_dir ${ProjectDir}results/tacotron-pml-x-163/gta/alignment/npy/ --eal_ckpt /home/miproj/4thyr.oct2018/je369/results/tacotron-schedsamp-pml-x-163/model.ckpt-150000


# python train.py --variant tacotron_pml_x_locsens --base_dir /scratch/je369/tacotron --name tacotron-pml-x-163-eal-scratch --log_dir /scratch/qd212/tacotron/results --num_steps 150000 --tacotron_input 163-lj-training/train.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --slack_url ${URL} --eal_dir ${ProjectDir}results/tacotron-pml-x-163/gta/alignment/npy/
# --restore_step 90000

# python train.py --variant tacotron_bk2orig --base_dir /scratch/je369/tacotron --name tacotron-bk2orig-eal-scratch --log_dir ${ProjectDir}results --num_steps 150000 --tacotron_input 163-lj-training/train.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --slack_url ${URL} --eal_dir ${ProjectDir}results/tacotron-pml-x-163/gta/alignment/npy/


# python train.py --variant tacotron_pml_x_locsens --base_dir /scratch/je369/tacotron --name tacotron-pml-x-163-eal-align --log_dir ${ProjectDir}results --num_steps 150000 --tacotron_input 163-lj-training/train.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --slack_url ${URL} --eal_dir ${ProjectDir}results/tacotron-pml-x-163/gta/alignment/npy/ --eal_trainAlign --eal_ckpt /scratch/qd212/tacotron/results/logs-tacotron-pml-x-163-eal-scratch/model.ckpt-120000

# python train.py --variant tacotron_pml_x_locsens --base_dir /scratch/je369/tacotron --name tacotron-pml-x-163-eal-joint1 --log_dir ${ProjectDir}results --num_steps 10000 --tacotron_input 163-lj-training/train.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --slack_url ${URL} --eal_dir ${ProjectDir}results/tacotron-pml-x-163/gta/alignment/npy/ --eal_trainJoint --eal_alignScale 1 --eal_ckpt ${ProjectDir}results/logs-tacotron-pml-x-163-eal-align/model.ckpt-5000

# python train.py --variant tacotron_pml_x_locsens --base_dir /scratch/je369/tacotron --name tacotron-pml-x-163-eal-joint1000 --log_dir ${ProjectDir}results --num_steps 10000 --tacotron_input 163-lj-training/train.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --slack_url ${URL} --eal_dir ${ProjectDir}results/tacotron-pml-x-163/gta/alignment/npy/ --eal_trainJoint --eal_alignScale 1000 --eal_ckpt ${ProjectDir}results/logs-tacotron-pml-x-163-eal-align/model.ckpt-5000

# python train.py --variant tacotron_pml_x_locsens --base_dir /scratch/je369/tacotron --name tacotron-pml-x-163-eal-joint50-scratch --log_dir ${ProjectDir}results --num_steps 150000 --tacotron_input 163-lj-training/train.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --slack_url ${URL} --eal_dir ${ProjectDir}results/tacotron-pml-x-163/gta/alignment/npy/ --eal_trainJoint --eal_alignScale 50 --restore_step 96000


python train.py --variant tacotron_pml_x_locsens --base_dir /scratch/qd212/tacotron --name tacotron-pml-x-163-eal-joint50-scratch-merlin --log_dir ${ProjectDir}results --num_steps 150000 --tacotron_input 163-lj-training/train_merlin.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --slack_url ${URL} --eal_dir ${ProjectDir}results/tacotron-pml-x-163-merlin/gta/alignment/npy/ --eal_trainJoint --eal_alignScale 50 --restore_step 107000








# 3.3 move the logs and results back to the home directory
# cd ${ProjectDir}/scripts/
# python move_log_dir.py --name $NAME

# # 4 for running on grid: remove data from air (optional)
# cd ${ProjectDir}/scripts/
# # python remove_training_data.py --train_dir 163-lj-training
