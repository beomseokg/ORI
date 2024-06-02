## Online Relational Inference for Evolving Multi-agent Interacting Systems ##
This is a temporary repository for NeurIPS submission number 14488.
It mainly includes ORI with MPM (*Neural Relational Inference with Efficient Message Passing Mechanisms* accepted by AAAI 2021. [arXiv](https://arxiv.org/pdf/2101.09486))

## Setup Environment
```bash
conda create -n ori python=3.8
conda activate ori
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchdiffeq==0.2.3
pip install torch_geometric==2.5.0
pip install torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-1.9.1+cu111.html
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.9.1+cu111.html
pip install torch-sparse==0.6.10 -f https://data.pyg.org/whl/torch-1.9.1+cu111.html
pip install tqdm==4.66.2
pip install matplotlib==3.0.3
pip install scikit-learn==1.3.2
```


## Setup Datasets

Data generation is primarily done by generate_dataset.py in /data.

To generate evolving interaction datasets (fixed parameter and no switching dynamics):
```bash
python generate_dataset.py --simulation springs --mode interaction
python generate_dataset.py --simulation springs --mode interaction
```

To generate evolving interaction + parameter datasets (variable parameter and no switching dynamics):
```bash
python generate_dataset.py --simulation springs --mode parameter
python generate_dataset.py --simulation springs --mode parameter
```

To generate evolving interaction + dynamics datasets (fixed parameter and switching dynamics), evolving interaction datasets should be prepared first.

## Run Experiments
Please check ```os.environ["CUDA_VISIBLE_DEVICES"]``` in NRI/train.py and NRI-MPM/run.py to select device to run experiments.

Current args are set to enable Trajectory Mirror and AdaRelation by default. Ablation studies can be perfomed by turning off these arguments.

For MPM,
```bash
cd NRI-MPM
python run.py --dyn springs # evolving interaction
python run.py --dyn charged # evolving interaction
python run.py --dyn springs_var # evolving interaction + parameter
python run.py --dyn charged_var # evolving interaction + parameter
python run.py --dyn mixed # evolving interaction + dynamics
```

For NRI,
```bash
cd NRI
python train.py --dyn springs # evolving interaction
python train.py --dyn charged # evolving interaction
python train.py --dyn springs_var # evolving interaction + parameter
python train.py --dyn charged_var # evolving interaction + parameter
python train.py --dyn mixed # evolving interaction + dynamics
```


## Acknowledgements

 - [NRI](https://github.com/ethanfetaya/NRI/tree/master)
 - [MPM](https://github.com/hilbert9221/NRI-MPM)

