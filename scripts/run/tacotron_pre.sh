# 0 preparation
NAME=tacotron-nick #////////////////////////////////////TBC

# 1 for running on grid: copy data to air
cd /home/miproj/4thyr.oct2018/je369/workspace/scripts
python check_remove_data.py
python check_move_data.py

# 2 preparation for the main code

# 2.1 for running on grid: go to right directory
cd /home/miproj/4thyr.oct2018/je369/workspace/implementations/tacotron

# 2.2 set up env
. /home/miproj/4thyr.oct2018/je369/.bashrc # we need to add conda commands to the path by running bashrc setup
conda info -e # print list of current environments for debug
conda activate /home/miproj/4thyr.oct2018/je369/workspace/implementations/tacotron/venv/
export PATH=${PATH}:/usr/local/cuda/bin

# 2.3 run the preprocess script
which python
python preprocess.py --base_dir /scratch/je369/tacotron/ --dataset nick

# 3 for running on grid: remove data from air (optional)
cd /home/miproj/4thyr.oct2018/je369/workspace/scripts
python check_remove_data.py
