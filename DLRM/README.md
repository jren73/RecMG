## DLRM 

This repository is RecMG's implementation with industrial level DLRM, and we use docker containers to increase reproducibility for users.
Docker image address for ReMG with industrial DLRM:
[DockerHub] https://hub.docker.com/repository/docker/binkma/rec_mg/general

### Download 

```
docker pull binkma/rec_mg
```

### Start

```
docker run -it --gpus=all -v /datapath:/workspace/data dlrm_fbgemm
```

### Setup
```
source /workspace/.bashrc
conda activate rec
```
### Rebuild FBGEMM

```
cd /workspace/FBGEMM/fbgemm_gpu
python setup.py install
```

### Rebuild torchrec

```
cd /workspace/torchrec
python setup.py install
```

### Run inference
```
torchx run -s local_cwd dist.ddp -j 1x1 --script dlrm_main.py
```

