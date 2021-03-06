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

########################### taco-pml trained with gta
# NAME=tacotron-pml-x-163-asup
NAME=tacotron-pml-x-163

# python synthesize.py --variant tacotron_pml_x_locsens --mode synthesis --gta True --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --text_list ${dataDir}/163-lj-training/test.txt --dataset train --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint /home/miproj/4thyr.oct2018/je369/results/tacotron-schedsamp-pml-x-163/model.ckpt-150000

# python synthesize.py --variant tacotron_pml_x_locsens --mode synthesis --eal True --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --text_list ${dataDir}/163-lj-training/test.txt --dataset train --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint /home/miproj/4thyr.oct2018/je369/results/tacotron-schedsamp-pml-x-163/model.ckpt-150000

# python synthesize.py --variant tacotron_pml_x_locsens --mode eval --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --text_list ${ProjectDir}/tests/sentences.txt --dataset train --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint /home/miproj/4thyr.oct2018/je369/results/tacotron-schedsamp-pml-x-163/model.ckpt-150000

# python synthesize.py --variant tacotron_pml_x_locsens --mode eval --base_dir /scratch/qd212/tacotron --output_dir ${ProjectDir}/results/${NAME}-CDI/ --training_dir 163-lj-training/ --text_list ${ProjectDir}/tests/sentences_10.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint /home/miproj/4thyr.oct2018/je369/results/tacotron-schedsamp-pml-x-163/model.ckpt-150000

# gen all alignments
# python synthesize.py --variant tacotron_pml_x_locsens --mode alignment --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --text_list ${dataDir}/163-lj-training/test.txt --dataset test --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint /home/miproj/4thyr.oct2018/je369/results/tacotron-schedsamp-pml-x-163/model.ckpt-150000


########################### taco-pml trained with gta and then scheduled sampling
# new check point
NAME=tacotron-schedsamp-pml-x-163
CKPT=/home/dawna/tts/qd212/models/tacotron/results/tacotron-schedsamp-pml-x-163/model.ckpt-240000
# CKPT=/home/miproj/4thyr.oct2018/je369/results/tacotron-schedsamp-pml-x-163/model.ckpt-150000

# EAL
# python synthesize.py --variant tacotron_pml_x_locsens --mode synthesis --eal True --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --text_list ${dataDir}/163-lj-training/test.txt --dataset train --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${CKPT}



########################### taco-pml trained with eal
# train with EAL
NAME=tacotron-pml-x-163-eal-scratch

# python synthesize.py --variant tacotron_pml_x_locsens --mode synthesis --eal True --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --text_list ${dataDir}/163-lj-training/test.txt --dataset train --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint /home/miproj/4thyr.oct2018/je369/results/tacotron-schedsamp-pml-x-163/model.ckpt-150000 --checkpoint_eal /scratch/qd212/tacotron/results/logs-${NAME}/model.ckpt-120000

# python synthesize.py --variant tacotron_pml_x_locsens --mode eval --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --text_list ${ProjectDir}/tests/sentences.txt --dataset train --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint /home/miproj/4thyr.oct2018/je369/results/tacotron-schedsamp-pml-x-163/model.ckpt-150000 --checkpoint_eal /scratch/qd212/tacotron/results/logs-${NAME}/model.ckpt-120000


NAME=tacotron-pml-x-163-eal-align
# python synthesize.py --variant tacotron_pml_x_locsens --mode eval --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --text_list ${ProjectDir}/tests/sentences.txt --dataset train --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${ProjectDir}/results/logs-${NAME}/model.ckpt-5000

# python synthesize.py --variant tacotron_pml_x_locsens --mode synthesis --eal True --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --text_list ${ProjectDir}/tests/sentences.txt --dataset test --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint /home/miproj/4thyr.oct2018/je369/results/tacotron-schedsamp-pml-x-163/model.ckpt-150000 --checkpoint_eal ${ProjectDir}/results/logs-${NAME}/model.ckpt-5000


NAME=tacotron-pml-x-163-eal-joint1000
# python synthesize.py --variant tacotron_pml_x_locsens --mode eval --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}-10k/ --training_dir 163-lj-training/ --text_list ${ProjectDir}/tests/sentences.txt --dataset train --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${ProjectDir}/results/logs-${NAME}/model.ckpt-10000


NAME=tacotron-pml-x-163-eal-joint50-scratch
# python synthesize.py --variant tacotron_pml_x_locsens --mode eval --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --text_list ${ProjectDir}/tests/sentences.txt --dataset train --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${ProjectDir}/results/logs-${NAME}/model.ckpt-85000
# ${ProjectDir}/results/logs-${NAME}/model.ckpt-145000 overfit!!!

# python synthesize.py --variant tacotron_pml_x_locsens --mode synthesis --eal True --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --text_list ${ProjectDir}/tests/sentences.txt --dataset train --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint /home/miproj/4thyr.oct2018/je369/results/tacotron-schedsamp-pml-x-163/model.ckpt-150000 --checkpoint_eal ${ProjectDir}/results/logs-${NAME}/model.ckpt-85000


########################### taco-pml trained with eal, online
dataDir=/scratch/qd212/tacotron
NAME=tacotron-pml-eal-joint50-merlin-online

# eal model
# python synthesize.py --variant tacotron_pml_x_locsens_online --mode eval --online --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --text_list ${ProjectDir}/tests/sentences.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${ProjectDir}/results/logs-${NAME}/model.ckpt-200000

# python synthesize.py --variant tacotron_pml_x_locsens_online --mode synthesis --eal True --online --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --dataset test_merlin --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${ProjectDir}/results/logs-${NAME}/model.ckpt-30000

# gta model
# python synthesize.py --variant tacotron_pml_x_locsens_online --mode eval --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/gta --training_dir 163-lj-training/ --text_list ${ProjectDir}/tests/sentences.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${ProjectDir}/results/logs-${NAME}/model.ckpt-30000

# python synthesize.py --variant tacotron_pml_x_locsens_online --mode synthesis --gta True --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --dataset test_merlin --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${ProjectDir}/results/logs-${NAME}/model.ckpt-30000

# NAME=tacotron-pml-eal-joint50-merlin-online-tfpre
NAME=tacotron-pml-eal-joint50-merlin-online-tfpre-tflock

# eal model
# python synthesize.py --variant tacotron_pml_x_locsens_online --mode eval --online --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --text_list ${ProjectDir}/tests/sentences.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${ProjectDir}/results/logs-${NAME}/model.ckpt-85000





########################### taco-pml CDI: clean data initiaitve
dataDir=/scratch/qd212/tacotron
NAME=tacotron-pml-x-163-merlin
# python synthesize.py --variant tacotron_pml_x_locsens --mode eval --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --text_list ${ProjectDir}/tests/sentences.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${ProjectDir}/results/logs-${NAME}/model.ckpt-244000
# 244000 200000 197000 150000 101000 66000

# python synthesize.py --variant tacotron_pml_x_locsens --mode synthesis --gta True --base_dir /scratch/qd212/tacotron --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --dataset train_merlin --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${ProjectDir}/results/logs-${NAME}/model.ckpt-75000

# gen all alignments
# python synthesize.py --variant tacotron_pml_x_locsens --mode alignment --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --dataset train_merlin --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${ProjectDir}/results/logs-${NAME}/model.ckpt-75000

NAME=tacotron-pml-x-163-eal-joint50-scratch-merlin
# python synthesize.py --variant tacotron_pml_x_locsens --mode synthesis --eal True --base_dir /scratch/qd212/tacotron --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --dataset train_merlin --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${ProjectDir}/results/logs-tacotron-pml-x-163-merlin/model.ckpt-75000  --checkpoint_eal ${ProjectDir}/results/logs-${NAME}/model.ckpt-75000

# python synthesize.py --variant tacotron_pml_x_locsens --mode eval --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --text_list ${ProjectDir}/tests/sentences.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${ProjectDir}/results/logs-${NAME}/model.ckpt-200000 # 75000 101000 150000 250000


NAME=tacotron-pml-x-163-eal-joint50-scratch-mer
# python synthesize.py --variant tacotron_pml_x_locsens --mode synthesis --eal True --base_dir /scratch/qd212/tacotron --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --dataset train_merlin --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint /home/miproj/4thyr.oct2018/je369/results/tacotron-schedsamp-pml-x-163/model.ckpt-150000 --checkpoint_eal ${ProjectDir}/results/logs-${NAME}/model.ckpt-75000

# python synthesize.py --variant tacotron_pml_x_locsens --mode eval --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --text_list ${ProjectDir}/tests/sentences.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${ProjectDir}/results/logs-${NAME}/model.ckpt-75000


########################### taco-pml-cmudict
dataDir=/scratch/qd212/tacotron

# GTA
NAME=tacotron-pml-merlin-cmudict
# python synthesize.py --variant tacotron_pml_x_locsens --mode synthesis --gta True --base_dir ${dataDir} --output_dir ${ProjectDir}/results/bkup/${NAME}/ --training_dir 163-lj-training/ --dataset test_merlin --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd,use_cmudict=True" --checkpoint ${ProjectDir}/results/bkup/logs-${NAME}/model.ckpt-92000

# python synthesize.py --variant tacotron_pml_x_locsens --mode eval --base_dir ${dataDir} --output_dir ${ProjectDir}/results/bkup/${NAME}/ --training_dir 163-lj-training/ --text_list ${ProjectDir}/tests/sentences.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd,use_cmudict=True" --checkpoint ${ProjectDir}/results/bkup/logs-${NAME}/model.ckpt-92000

# python synthesize.py --variant tacotron_pml_x_locsens --mode eval --base_dir ${dataDir} --output_dir ${ProjectDir}/results/bkup/${NAME}-asup/ --training_dir 163-lj-training/ --text_list ${ProjectDir}/tests/sentences_asup.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${ProjectDir}/results/bkup/logs-${NAME}/model.ckpt-92000

# python synthesize.py --variant tacotron_pml_x_locsens --mode eval --base_dir ${dataDir} --output_dir ${ProjectDir}/results/bkup/${NAME}-asup-fr/ --training_dir 163-lj-training/ --text_list ${ProjectDir}/tests/sentences_asup_fr.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${ProjectDir}/results/bkup/logs-${NAME}/model.ckpt-92000

# gen all alignments
# python synthesize.py --variant tacotron_pml_x_locsens --mode alignment --base_dir ${dataDir} --output_dir ${ProjectDir}/results/bkup/${NAME}/ --training_dir 163-lj-training/ --dataset train_merlin --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${ProjectDir}/results/bkup/logs-${NAME}/model.ckpt-92000


########################### taco-pml-phone
dataDir=/scratch/qd212/tacotron

# GTA
NAME=tacotron-pml-merlin-phone
# python synthesize.py --variant tacotron_pml_x_locsens --mode synthesis --gta True --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --dataset test_merlin_phone --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${ProjectDir}/results/logs-${NAME}/model.ckpt-82000

# python synthesize.py --variant tacotron_pml_x_locsens --mode eval --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --text_list ${ProjectDir}/tests/sentences_phone.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${ProjectDir}/results/logs-${NAME}/model.ckpt-150000
# 82000 107000


# GTA
NAME=tacotron-pml-merlin-phone-punc
# python synthesize.py --variant tacotron_pml_x_locsens --mode synthesis --gta True --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --dataset test_merlin_phone --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${ProjectDir}/results/logs-${NAME}/model.ckpt-27000

# python synthesize.py --variant tacotron_pml_x_locsens --mode eval --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --text_list ${ProjectDir}/tests/sentences_phone_punc.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${ProjectDir}/results/logs-${NAME}/model.ckpt-200000 # 76000 101000 130000 170000 250000

# gen all alignments
# python synthesize.py --variant tacotron_pml_x_locsens --mode alignment --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --dataset train_merlin_phone_punc --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${ProjectDir}/results/logs-${NAME}/model.ckpt-130000


NAME=tacotron-pml-eal-joint50-scratch-merlin-phone-punc
# python synthesize.py --variant tacotron_pml_x_locsens --mode synthesis --eal True --base_dir /scratch/qd212/tacotron --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --dataset train_merlin_phone_punc --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${ProjectDir}/results/logs-tacotron-pml-merlin-phone-punc/model.ckpt-130000 --checkpoint_eal ${ProjectDir}/results/logs-${NAME}/model.ckpt-200000

# python synthesize.py --variant tacotron_pml_x_locsens --mode eval --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --text_list ${ProjectDir}/tests/sentences_phone_punc.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${ProjectDir}/results/logs-${NAME}/model.ckpt-200000 # 40000 60000

# python synthesize.py --variant tacotron_pml_x_locsens --mode eval --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/fr --training_dir 163-lj-training/ --text_list ${ProjectDir}/tests/sentences_phone_punc_fr.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${ProjectDir}/results/logs-${NAME}/model.ckpt-200000





###########################
########################### taco classic trained with gta
# gen all alignments
NAME=tacotron-bk2orig
# python synthesize.py --variant tacotron_bk2orig --mode alignment --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --text_list ${dataDir}/163-lj-training/test.txt --dataset test --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${ProjectDir}/results/logs-${NAME}/model.ckpt-120000

NAME=tacotron-bk2orig-eal-scratch
python synthesize.py --variant tacotron_bk2orig --mode eval --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --text_list ${ProjectDir}/tests/sentences.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${ProjectDir}/results/logs-${NAME}/model.ckpt-150000



# # 3.3 move the logs and results back to the home directory
# cd /home/dawna/tts/qd212/models/tacotron-master/scripts/
# python move_log_dir.py --name $NAME

# # 4 for running on grid: remove data from air (optional)
# cd /home/miproj/4thyr.oct2018/je369/workspace/implementations/tacotron/scripts
# # python remove_training_data.py --train_dir 163-lj-training
