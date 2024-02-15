# InfoCon
This is the official repository for: **[InfoCon: Concept Discovery with Generative and Discriminative Informativeness](https://openreview.net/forum?id=g6eCbercEc&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2024%2FConference%2FAuthors%23your-submissions))**

<p align="center">
  <img src='github_teaser/infocon.png' width="700"><br>
</p>
<p align="center">
  <img src='github_teaser/gg_and_dg.jpg' width="700"><br>
</p>

## Environment
### Hardware & OS

64 CPUs, NVIDIA GeForce RTX 3090 (NVIDIA-SMI 530.30.02, Driver Version: 530.30.02, CUDA Version: 12.1), Ubuntu 20.04.6 LTS (GNU/Linux 5.15.0-83-generic x86_64)
### Installation

```
conda create -n infocon python=3.9
source activate infocon
pip install -r requirements.txt
```

## Usage

### Training

### Labeling

### CoTPC Evaluation

<div style='display: none'>

## CoTPC-main/
relates to CoTPC downstream policies.
* **data**: ManiSkill2 data-set.
* **maniskill2_patches**: Some patching code in ManiSkill2 for CoTPC logs. Refer to CoTPC GitHub Repo for details...
* **scripts**: bash scripts for CoTPC training and evaluation.
* **src**: src code related to CoTPC policies.
* **save_model**: checkpoints of CoTPC policies.

## src/
includes the codes of InfoCon, where
* **modules** includes the used DNN modules
  * **GPT.py**: Transformers used in InfoCon
  * **VQ.py**: VQ-VAE used in InfoCon. It is a little bit different from vanilla VQ-VAE. We've tried many kinds of design. Currently we are using **VQClassifierNNTime**.
  * **module_util.py**: Other modules, like some MLPs, time step embedding modules.
  * (currently other source file are unused)
* **autocot.py**: construct different modules into whole InfoCon. Refer to it for the main pipeline of InfoCon.
* **data.py**: load data.
* **vec_env.py**: Relate to ManiSkill2. Vectorize Environments.
* **train.py**: python scripts for InfoCon training.
* **path.py**: log of data and checkpoint file paths.
* **callbacks.py**: Customized Callbacks for PyTorch Lightning training of InfoCon.
* **label.py**: python scripts for labeling key states. Labeled out key states will be stored as .txt file in **CoTPC-main/data/$TASK_DIR$**.
* **his.py**: calculate Human Intuition Score (HIS) when given labeled out key states.
* **util.py**: other modules and functions.

</div>


