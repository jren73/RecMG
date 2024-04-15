## DLRM 

This repository is RecMG's implementation with industrial level DLRM, and we use docker containers to increase reproducibility for users.


### Download 

```
docker pull binkma/rec_mg
```

### Start

```
docker run -it --gpus=all -v /path\_to\_data:/workspace/data dlrm\_fbgemm
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

