# InfoCon
Use Pytorch Lightning in some source codes.

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


