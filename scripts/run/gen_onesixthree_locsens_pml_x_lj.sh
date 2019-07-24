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
# cd /home/miproj/4thyr.oct2018/je369/workspace/implementations/tacotron/scripts
# python remove_training_data.py --train_dir 163-lj-training
# python check_move_data.py

# # 2 preparation for the main code

# # 2.1 for running on grid: go to right directory
# cd /home/miproj/4thyr.oct2018/je369/workspace/implementations/tacotron

# # 2.2 run the preprocess script
# which python
# python preprocess.py --base_dir /scratch/je369/tacotron/ --dataset ljspeech --output 163-lj-training --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --validation_size 150 --test_size 150 # ljspeech used as dataset key

# # 2.3 remove the original data
# cd /home/miproj/4thyr.oct2018/je369/workspace/implementations/tacotron/scripts
# python remove_source_data.py

# # 3 train the model

# # 3.1 create the directory needed for logging
# cd /home/dawna/tts/qd212/models/tacotron-master/scripts/
# python create_log_dir.py

# 3.2 run the train scheme
dataDir=/scratch/je369/tacotron
cd /home/dawna/tts/qd212/models/tacotron/

NAME=tacotron-pml-x-163-asup

# python synthesize.py --variant tacotron_pml_x_locsens --mode synthesis --gta True --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --text_list ${dataDir}/163-lj-training/test.txt --dataset train --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint /home/miproj/4thyr.oct2018/je369/results/tacotron-schedsamp-pml-x-163/model.ckpt-150000

# python synthesize.py --variant tacotron_pml_x_locsens --mode synthesis --eal True --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --text_list ${dataDir}/163-lj-training/test.txt --dataset train --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint /home/miproj/4thyr.oct2018/je369/results/tacotron-schedsamp-pml-x-163/model.ckpt-150000

# python synthesize.py --variant tacotron_pml_x_locsens --mode eval --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --text_list ${ProjectDir}/tests/sentences.txt --dataset train --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint /home/miproj/4thyr.oct2018/je369/results/tacotron-schedsamp-pml-x-163/model.ckpt-150000

# gen all alignments
# python synthesize.py --variant tacotron_pml_x_locsens --mode alignment --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --text_list ${dataDir}/163-lj-training/test.txt --dataset train --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint /home/miproj/4thyr.oct2018/je369/results/tacotron-schedsamp-pml-x-163/model.ckpt-150000


# new check point
NAME=tacotron-schedsamp-pml-x-163
CKPT=/home/dawna/tts/qd212/models/tacotron/results/tacotron-schedsamp-pml-x-163/model.ckpt-240000
# CKPT=/home/miproj/4thyr.oct2018/je369/results/tacotron-schedsamp-pml-x-163/model.ckpt-150000

# EAL
# python synthesize.py --variant tacotron_pml_x_locsens --mode synthesis --eal True --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --text_list ${dataDir}/163-lj-training/test.txt --dataset train --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${CKPT}


# train with EAL
NAME=tacotron-pml-x-163-eal-scratch

# python synthesize.py --variant tacotron_pml_x_locsens --mode synthesis --eal True --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --text_list ${dataDir}/163-lj-training/test.txt --dataset train --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint /home/miproj/4thyr.oct2018/je369/results/tacotron-schedsamp-pml-x-163/model.ckpt-150000 --checkpoint_eal /scratch/qd212/tacotron/results/logs-${NAME}/model.ckpt-120000

# python synthesize.py --variant tacotron_pml_x_locsens --mode eval --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --text_list ${ProjectDir}/tests/sentences.txt --dataset train --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint /home/miproj/4thyr.oct2018/je369/results/tacotron-schedsamp-pml-x-163/model.ckpt-150000 --checkpoint_eal /scratch/qd212/tacotron/results/logs-${NAME}/model.ckpt-120000



# # 3.3 move the logs and results back to the home directory
# cd /home/dawna/tts/qd212/models/tacotron-master/scripts/
# python move_log_dir.py --name $NAME

# # 4 for running on grid: remove data from air (optional)
# cd /home/miproj/4thyr.oct2018/je369/workspace/implementations/tacotron/scripts
# # python remove_training_data.py --train_dir 163-lj-training
