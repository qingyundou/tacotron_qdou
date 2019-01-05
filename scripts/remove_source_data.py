import os
import subprocess

def checkDir(directory):
    if os.path.exists(directory):
        return True
    else:
        return False
    
def runCMD(cmd):
    df = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = df.communicate()
    return output,err

# (1) remove data from air

# (1.1) loop through all the datasets
for dataset in ['Nick', 'LJSpeech-1.1']:
    # (1.1.1) check if the directory is already there
    directory = f'/scratch/je369/tacotron/{dataset}'
    FLAG_DATA_ALREADY = checkDir(directory)

    # (1.1.2) if not already there, no need to remove
    if FLAG_DATA_ALREADY:
        print(f'Data {dataset} found on air, removing ...')
        cmd = 'rm -r ' + directory
        output,err = runCMD(cmd)
        print('Output: \n'+str(output))
        print('Err: \n'+str(err))
        print(f'Removing {dataset} complete!')
    else:
        print('Data not found on air, no need to remove')
