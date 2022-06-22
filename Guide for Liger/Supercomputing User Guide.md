<link rel="stylesheet" type="text/css" href="auto-number-title.css" />

# Supercomputer User Guide
## Basic linux operations
### List
```
ll
ls
```
### display where now
```
pwd
```
### Create file
```
touch test.txt
```
### Remove file
```
rm test.txt
```
### Open file
```
cat run.sh
```
### Navigate
move to [folder name]
```
cd [foldername]
```
move to last folder
```
cd..
```
move to home
```
cd
```

## Liger
### Access to liger
```
ssh liger
```
or
```
ssh arvgxwnfe@liger.ec-nantes.fr
```
### Storage on liger
Here are 3 distinct disk spaces available for each project: HOME, DATA, SCRATCH.

$HOME : This is the home directory during an interactive connection login by SSH. This space is intended for frequently-used small-sized files such as the shell environment files, the tools, and potentially the sources and libraries if they have a reasonable size. 
```
cd $HOME
```
$DATADIR: This is a permanent project storage space to store large-sized files (max 100GB per project) for use during batch executions related to the project: very large source files, libraries, data files, executable files, result files and submission scripts.
```
cd $DATADIR
```
$SCRATCHDIR : This is a semi-permanent work and large storage space. You can run your code here and transfer the result to `DATA`, because this space is not a backup space.
```
cd $SCRATCHDIR
```
### Run code on liger
#### Prepare `run.py` and `job.sl`
On your host computer prepare `run.py` and `job.sl`. Add `#!/usr/bin/env python3` at the beginning of `run.py`. 

Refer to the following `job.sl` template to create a `job.sl`.
```
#!/bin/bash
#SBATCH --job-name=DRAMAinPT       # name of job
#SBATCH --partition=gpus
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6            # number of cores per task setting the MEMORY,
                                     # mem = 32GB / 48 * cpus-per-task
#SBATCH --hint=nomultithread         # hyperthreading is deactivated
#SBATCH --time=04:00:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=Result-DRAMAinPT-%j.out         # name of output file
#SBATCH --error=Error_DRAMAinPT-%j.out          # name of error file (here, appended with the output file)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lixinyi9812@gmail.com

# Setting the container image location
BASE_DIR=/scratch/arvgxwnfe/DRAMAinPT # Directory containing AI-related containers in Liger
IMAGE=pytorch-dramainpt.sif           # Container image

# cleans out the modules loaded in interactive and inherited by default
module purge

# load singularity: container engine to execute the image
module load singularity

# bind and run your program in the container
singularity exec --nv -B ./: /workspace/shared \
/scratch/arvgxwnfe/DRAMAinPT/pytorch-dramainpt.sif python3 /workspace/shared/DRAMAinPT_final.py
```
Note that the volume(It allows the host to share its own file system with the container) `/workspace/shared` in container is bind with the folder where `CelebA.py` is.
`-B` means bind. In this example, the host directory `./` in liger will be mounted as `/workspace/shared` in the container. Then use `/scratch/arvgxwnfe/DRAMAinPT/pytorch-dramainpt.sif` container, let `python3` be the compiler and run `/workspace/shared/DRAMAinPT_final.py`.


#### Transfer `run.py` and `job.sl` to liger
Go to where these two files are:
```
cd [folderpath]
```
Transfer files
```
scp run.py liger:/scratch/arvgxwnfe/
scp job.sl liger:/scratch/arvgxwnfe/
```
or
```
scp -r [foldername]` liger:/scratch/arvgxwnfe/
```
Note that `arvgxwnfe` is user account name. `-r` means copy the whole folder.
Every time you change the code, you should transfer it again to liger.

Then enter the folder transfered to liger:
```
ssh liger
cd /scratch/arvgxwnfe/[foldername]
```
Run slurm job / submit job
```
sbatch job.sl
```
See sequence
```
Mysqueue
```
Check the realtime file(list latest info)
```
tail -f [filename]
```
See result
```
sacct -X
```
After obtaining the result file, don't forget to store it in `$DATA` or your host computer.
Operation at the host: send file from liger to host(. means where I am now on host):
```
scp liger:/scratch/arvgxwnfe/[result file] .
```
Operation at liger: Send file from `$SCRATCH` to `$DATA`:
```
cp [result file] /data/OG2102040
```
`scp`  for transfer between host and liger. `cp` for transfer inside liger or host.
### Modify directly on liger
In some case you want to modify files directly on liger:
```
nano [filename]
```
Enter edit mode
```
^O 
^enter
```
### List all gpu on liger
```
sinfo -o "%20N  %10c  %10m  %25f  %10G "
```

### Get real-time insight on used GPU
```
ssh turining04
watch -n 5 nvidia-smi
```
Use `ctrl+c` to quit.

## Customize an image
In order to run code on liger, we have to make an image of the environment configuration of `run.py` and send it to liger.
### Obtain requirements.txt file in conda environment
First you should configure the environment of `run.py` and make sure it works well. Then
go to Anaconda Prompt and activate your virtual environment to get the `requirements.txt` file:
```
conda activate [env name]
pip list --format=freeze > requirements.txt
```
### Modify dockerfile on host computer
We use dockerfile to create an image.
The template dockerfile and requirements.txt can be download here: https://gitlab.in2p3.fr/ecn-collaborations/liger-ai-tools/-/tree/master/docker/tensorflow/2.6.1  
```
# Choose a base image from official dockerhub repository
FROM tensorflow/tensorflow:2.6.1-gpu

# vars definition
ENV DEBIAN_FRONTEND noninteractive

WORKDIR /workspace

# upgrade pip
RUN python3 -m pip install -U pip

COPY requirements.txt /workspace/

# install packages defined in fat_package_list.txt
RUN pip install -r requirements.txt


ENTRYPOINT [ "bash" ]
```


### Using customized dockerfile to build an image
```
docker build -t [imagename] --file Dockerfile-fat .
```
Check your image
```
docker image list
```
Export and compress the image to a `tar.gz` file
```
docker save [imagename] | gzip > [imagename].tar.gz
```
Upload it to LIGER (don't forget to connect to Centrale's VPN!) :

```
scp [imagename].tar.gz <username>@liger.ec-nantes.fr:/scratch/<username>
```

###  Convert the image to SIF on liger
We need to convert the docker image to a `.sif` file that can be used by Singularity on liger.

Reserve a GPU :
```
srun --pty -p gpus --gres=gpu:1 --account=gpu-others --cpus-per-task=4 --hint=nomultithread --time=00:30:00 bash
```
SSH into the GPU. First, check on which server your job is running (in the second to last column):
```
Mysqueue
```
For example, if your job is running on `viz03`:
```
ssh -X viz03
```
Convert to a SIF image:
```
cd $SCRATCHDIR
module load singularity
export SINGULARITY_CACHEDIR=$SCRATCHDIR/singularity/cache
singularity build [imagename].sif docker-archive://[imagename].tar.gz
```
After a few moments, your `.sif` image is created. You can now use it, to do so submit a job with slurm and use `singularity exec --nv [imagename].sif <your command>`.
Don't forget to cancel the job once the image is converted! First, log out of the gpu server with `ctrl + d`. Then, run:
```
scancel <job id>
```
Note: the job id can be found with `Mysqueue` or `sacct -X`.
You may want move the image to a backed-up partition, like `$DATADIR`:
```
# LIPID is OG2102040 for us
mv $SCRATCHDIR/[imagename].sif $DATADIR/<LIPID>
```

## Use an existed image
There are some available image provided by ECN. If one of the images suits your `run.py`, you can use it directly instead of creating dockerfile by yourself.

Pull an image from ECN:
```
docker pull gitlab-registry.in2p3.fr/ecn-collaborations/liger-ai-tools/tensorflow-2.6.1-fat
```
Start a container instance on the base image
```
docker run --name [container_name] -idt [image_name]
```
or run it directly in Docker application.

Enter container
```
docker exec -it [container_name] /bin/bash
```
In container, you can install dependencies
```
pip install
```
Exit container
```
exit
```
Build your image
```
 docker commit -a 'author' -m 'instruction' [container_name] [image_name]
```
Compress image
```
docker save -o test_tar.tar [image_name]
```
The tar file can be finded in current folder.

## Manage image by docker
### List all image
```
docker image ls
```
### Run file(used for debug locally)
```
docker run -it --rm -v "$(pwd)":/workspace/shared gitlab-registry.in2p3.fr/ecn-collaborations/liger-ai-tools/tensorflow-2.6.1-fat
```
### Run code(used for debug locally)
```
docker run -it --rm -v "$(pwd)":/workspace/shared gitlab-registry.in2p3.fr/ecn-collaborations/liger-ai-tools/tensorflow-2.6.1-fat -c shared/run.sh
```
### Navigate to container
cd /softs/singularity/containers/ai














