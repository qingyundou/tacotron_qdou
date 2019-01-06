import os
import subprocess

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

# (1) copy data to air
tgt_directory = '/scratch/je369/tacotron/'

# (1.1) copy Nick data to air
nick_directory = 'Nick'
text_src_dataset = '/home/astra2/tts/data/txt/Eng/NickDNNDemo'
text_target_dir = 'txt'
wav_src_dataset = '/home/astra2/tts/data/wav/Eng/NickDNNDemo'
wav_target_dir = 'wav'
pml_src_dataset = '/home/miproj/4thyr.oct2018/je369/tools/merlin_je369/exp/nick/data/nn_cmp'
pml_target_dir = 'pml'

# (1.1.1) check root Nick directory is created
t = os.path.join(tgt_directory, nick_directory)
checkMakeDir(t)

# (1.1.2) copy Nick text to air
t = os.path.join(tgt_directory, nick_directory, text_target_dir)
checkMakeDir(t)
cmd = 'cp -r {s}/* {t}/'.format(s=text_src_dataset, t=t)
print('Moving Nick text with cmd: '+cmd)
output,err = runCMD(cmd)
print('Output (Nick text): '+str(output))
print('Err (Nick text): '+str(err))

# (1.1.3) copy Nick wav files to air
t = os.path.join(tgt_directory, nick_directory, wav_target_dir)
checkMakeDir(t)
cmd = 'cp -r {s}/* {t}/'.format(s=wav_src_dataset, t=t)
print('Moving Nick audio with cmd: '+cmd)
output,err = runCMD(cmd)
print('Output (Nick audio): '+str(output))
print('Err (Nick audio): '+str(err))

# (1.1.4) copy Nick PML feature files to air
t = os.path.join(tgt_directory, nick_directory, pml_target_dir)
checkMakeDir(t)
cmd = 'cp -r {s}/* {t}/'.format(s=pml_src_dataset, t=t)
print('Moving Nick PML features with cmd: '+cmd)
output,err = runCMD(cmd)
print('Output (Nick PML): '+str(output))
print('Err (Nick PML): '+str(err))

# (1.1.5) message move is complete
print('Moving Nick data complete!')

# (1.2) copy LJ data to air
lj_directory = 'LJSpeech-1.1'
metadata_src_location = '/home/dawna/tts/data/LJSpeech-1.1/webData/metadata.csv'
metadata_target_file = 'metadata.csv'
wav_src_dataset = '/home/dawna/tts/data/LJSpeech-1.1/webData/wavs'
wav_target_dir = 'wavs'
pml_src_dataset = '/home/dawna/tts/data/LJSpeech-1.1/merlinData/nn_cmp'
pml_target_dir = 'pml'

# (1.2.1) check root LJ directory is created
t = os.path.join(tgt_directory, lj_directory)
checkMakeDir(t)

# (1.2.2) copy LJ metadata.csv file to air
t = os.path.join(tgt_directory, lj_directory, metadata_target_file)
cmd = 'cp {s} {t}'.format(s=metadata_src_location, t=t)
print('Moving LJ metadata.csv with cmd: '+cmd)
output,err = runCMD(cmd)
print('Output (LJ metadata.csv): '+str(output))
print('Err (LJ metadata.csv): '+str(err))

# (1.2.3) copy LJ wav files to air
t = os.path.join(tgt_directory, lj_directory, wav_target_dir)
checkMakeDir(t)
cmd = 'cp -r {s}/* {t}/'.format(s=wav_src_dataset, t=t)
print('Moving LJ audio with cmd: '+cmd)
output,err = runCMD(cmd)
print('Output (LJ audio): '+str(output))
print('Err (LJ audio): '+str(err))

# (1.2.4) copy LJ PML feature files to air
t = os.path.join(tgt_directory, lj_directory, pml_target_dir)
checkMakeDir(t)
cmd = 'cp -r {s}/* {t}/'.format(s=pml_src_dataset, t=t)
print('Moving LJ PML features with cmd: '+cmd)
output,err = runCMD(cmd)
print('Output (LJ PML): '+str(output))
print('Err (LJ PML): '+str(err))

# (1.2.5) message move is complete
print('Moving LJ data complete!')
