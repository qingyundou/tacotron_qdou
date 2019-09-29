# Attention forcing - speech synthesis


An implementation of attention forcing, in the context of speech synthesis, in TensorFlow.


## Background
Auto-regressive sequence-to-sequence models with attention mechanism have achieved state-of-the-art performance in many tasks such as machine translation and speech synthesis. These models can be difficult to train. The standard approach, teacher forcing, guides a model with reference output history during training. The problem is that the model is unlikely to recover from its mistakes during inference, where the reference output is replaced by generated output. Several approaches deal with this problem, largely by guiding the model with generated output history. To make training stable, these approaches often require a heuristic schedule or an auxiliary classifier. This paper introduces attention forcing, which guides the model with generated output history and reference attention. This approach can train the model to recover from its mistakes, in a stable fashion, without the need for a schedule or a classifier. In addition, it allows the model to generate output sequences aligned with the references, which can be important for cascaded systems like many speech synthesis systems.

Experiments on speech synthesis show that attention forcing yields significant performance gain. The frame-level model is described in [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/pdf/1703.10135.pdf); the code is in the current repo. The waveform-level model is described in [Hierarchical RNNs for Waveform-Level Speech Synthesis](https://ieeexplore.ieee.org/document/8639588) and [SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://arxiv.org/abs/1612.07837); the code is in this repo: https://github.com/qingyundou/sampleRNN_QDOU



## Quick Start

### Installing dependencies

1. Install Python 3.

2. Install the latest version of [TensorFlow](https://www.tensorflow.org/install/) for your platform. For better
   performance, install with GPU support if it's available. This code works with TensorFlow 1.3 and later.

3. Install requirements:
   ```
   pip install -r requirements.txt
   ```



### Training

*Note: you need at least 40GB of free disk space to train a model.*

1. **Download a speech dataset.**

   The following are supported out of the box:
    * [LJ Speech](https://keithito.com/LJ-Speech-Dataset/) (Public Domain)
    * [Blizzard 2012](http://www.cstr.ed.ac.uk/projects/blizzard/2012/phase_one) (Creative Commons Attribution Share-Alike)

   You can use other datasets if you convert them to the right format. See [TRAINING_DATA.md](TRAINING_DATA.md) for more info.


2. **Unpack the dataset into `~/tacotron`**

   After unpacking, your tree should look like this for LJ Speech:
   ```
   tacotron
     |- LJSpeech-1.1
         |- metadata.csv
         |- wavs
   ```

   or like this for Blizzard 2012:
   ```
   tacotron
     |- Blizzard2012
         |- ATrampAbroad
         |   |- sentence_index.txt
         |   |- lab
         |   |- wav
         |- TheManThatCorruptedHadleyburg
             |- sentence_index.txt
             |- lab
             |- wav
   ```

3. **Preprocess the data**
   ```
   python3 preprocess.py --dataset ljspeech
   ```
     * Use `--dataset blizzard` for Blizzard data

4. **Train a model**
    In general:
   ```
   python3 train.py
   ```

   Tunable hyperparameters are found in [hparams.py](hparams.py). You can adjust these at the command
   line using the `--hparams` flag, for example `--hparams="batch_size=16,outputs_per_step=2"`.
   Hyperparameters should generally be set to the same values at both training and eval time.
   The default hyperparameters are recommended for LJ Speech and other English-language data.
   See [TRAINING_DATA.md](TRAINING_DATA.md) for other languages.
   
   Example of training with teacher forcing:
   ```
   python train.py --variant tacotron_pml_x_locsens --base_dir /scratch/qd212/tacotron --name tacotron-pml-x-163-merlin --log_dir ${ProjectDir}results --num_steps 150000 --tacotron_input 163-lj-training/train_merlin.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd"
   ```
   
   Example of training with attention forcing:
   ```
   python train.py --variant tacotron_pml_x_locsens --base_dir /scratch/qd212/tacotron --name tacotron-pml-x-163-eal-joint50-scratch-merlin --log_dir ${ProjectDir}results --num_steps 150000 --tacotron_input 163-lj-training/train_merlin.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --slack_url ${URL} --eal_dir ${ProjectDir}results/tacotron-pml-x-163-merlin/gta/alignment/npy/ --eal_trainJoint --eal_alignScale 50
   ```


5. **Monitor with Tensorboard** (optional)
   ```
   tensorboard --logdir ~/tacotron/logs-tacotron
   ```

   The trainer dumps audio and alignments every 1000 steps. You can find these in
   `~/tacotron/logs-tacotron`.

6. **Synthesize from a checkpoint**
   In general, you can run [synthesize.py](tacotron/synthesize.py) at the command line:
   ```
   python3 synthesize.py --checkpoint ~/tacotron/logs-tacotron/model.ckpt-185000
   ```
   If you set the `--hparams` flag when training, set the same value here.
   
   Example of synthesizing in free running mode (with either a teacher forcing model or an attention forcing model):
   ```
   python synthesize.py --variant tacotron_pml_x_locsens --mode eval --base_dir ${dataDir} --output_dir ${ProjectDir}/results/${NAME}/ --training_dir 163-lj-training/ --text_list ${ProjectDir}/tests/sentences.txt --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd" --checkpoint ${ProjectDir}/results/logs-${NAME}/model.ckpt-75000
   ```


### Training on server

The machines `air208` and `air209` have CUDA 9.0 installed, which is required to run this code. To run on either machine, run the command:

```
qsub -M YOUREMAIL -m bea -S /bin/bash -l queue_priority=cuda-low,tests=0,mem_grab=0M,gpuclass=*,osrel=*,hostname=air209 /home/miproj/4thyr.oct2018/je369/workspace/implementations/tacotron/scripts/run/onesixthree_locsens_pml_x_lj.sh
```


## Notes and Common Issues

  * [TCMalloc](http://goog-perftools.sourceforge.net/doc/tcmalloc.html) seems to improve
    training speed and avoids occasional slowdowns seen with the default allocator. You
    can enable it by installing it and setting `LD_PRELOAD=/usr/lib/libtcmalloc.so`. With TCMalloc,
    you can get around 1.1 sec/step on a GTX 1080Ti.

  * You can train with [CMUDict](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) by downloading the
    dictionary to ~/tacotron/training and then passing the flag `--hparams="use_cmudict=True"` to
    train.py. This will allow you to pass ARPAbet phonemes enclosed in curly braces at eval
    time to force a particular pronunciation, e.g. `Turn left on {HH AW1 S S T AH0 N} Street.`

  * If you pass a Slack incoming webhook URL as the `--slack_url` flag to train.py, it will send
    you progress updates every 1000 steps.

  * Occasionally, you may see a spike in loss and the model will forget how to attend (the
    alignments will no longer make sense). Although it will recover eventually, it may
    save time to restart at a checkpoint prior to the spike by passing the
    `--restore_step=150000` flag to train.py (replacing 150000 with a step number prior to the
    spike). **Update**: a recent [fix](https://github.com/keithito/tacotron/pull/7) to gradient
    clipping by @candlewill may have fixed this.
    
  * During eval and training, audio length is limited to `max_iters * outputs_per_step * frame_shift_ms`
    milliseconds. With the defaults (max_iters=200, outputs_per_step=5, frame_shift_ms=12.5), this is
    12.5 seconds.
    
    If your training examples are longer, you will see an error like this:
    `Incompatible shapes: [32,1340,80] vs. [32,1000,80]`
    
    To fix this, you can set a larger value of `max_iters` by passing `--hparams="max_iters=300"` to
    train.py (replace "300" with a value based on how long your audio is and the formula above).
    
  * Here is the expected loss curve when training on LJ Speech with the default hyperparameters:
    ![Loss curve](https://user-images.githubusercontent.com/1945356/36077599-c0513e4a-0f21-11e8-8525-07347847720c.png)
