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

python spec2wav.py
