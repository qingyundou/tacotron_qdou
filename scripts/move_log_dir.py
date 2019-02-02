import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='')
args = parser.parse_args()

def checkDir(directory):
    if os.path.exists(directory):
        return True
    else:
        return False

def checkMakeDir(directory):
    if os.path.exists(directory):
        return True
    else:
        os.makedirs(directory)
        return False

def runCMD(cmd):
    df = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = df.communicate()
    return output,err

# (1) copy logs, checkpoints and alignments to cloud
base_dir = '/scratch/je369/results/'
log_directory = f'logs-{args.name}'
tgt_directory = f'/home/miproj/4thyr.oct2018/je369/results/{args.name}'

# (1.1.2) copy logs to cloud
s = os.path.join(base_dir, log_directory)
checkMakeDir(tgt_directory)
cmd = 'cp -r {s}/* {t}/'.format(s=s, t=tgt_directory)
print('Moving logs with cmd: '+cmd)
output, err = runCMD(cmd)
print('Output: '+str(output))
print('Err: '+str(err))

# (1.1.3) remove the logs from the device
# print('Removing logs directory...')
# cmd = 'rm -r ' + s
# output, err = runCMD(cmd)
# print('Output: \n'+str(output))
# print('Err: \n'+str(err))
# print('Removing complete!')

# (1.1.4) message move is complete
print('Moving output logs complete!')
