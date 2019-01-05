import os
import subprocess

def checkMakeDir(directory):
    if os.path.exists(directory):
        return True
    else:
        os.makedirs(directory)
        return False

# (1) create log directory
log_directory = '/scratch/je369/results/'
checkMakeDir(log_directory)
