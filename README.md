# Dataset Size Recovery from LoRA Weights

Official Pytorch implementation of Dataset Size Recovery from LoRA Weights paper.
<p align="center">
🌐 <a href="https://vision.huji.ac.il/dsire/" target="_blank">Project</a> | 📃 <a href="https://arxiv.org/abs/2406.19395" target="_blank">Paper</a><br> 🤗 <a href="https://huggingface.co/datasets/MoSalama98/LoRA-WiSE" target="_blank">Dataset</a> <br>
</p>

![](imgs/diagrama.gif)


> **Dataset Size Recovery from LoRA Weights**<br>
> Mohammad Salama, Jonathan Kahana, Eliahu Horwitz, Yedid Hoshen<br>
> <a href="https://arxiv.org/abs/2406.19395" target="_blank">https://arxiv.org/abs/2406.19395 <br>
>
>**Abstract:** Model inversion and membership inference attacks aim to reconstruct and verify the data which a model was trained on.
> However, they are not guaranteed to find all training samples as they do not know the size of the training set. In this paper, we introduce
> a new task: dataset size recovery, that aims to determine the number of samples used to train a model, directly from its weights. We then propose
> **DSiRe**, a method for recovering the number of images used to fine-tune a model, in the common case where fine-tuning uses LoRA.
> We discover that both the norm and the spectrum of the LoRA matrices are closely linked to the fine-tuning dataset size; we leverage this
> finding to propose a simple yet effective prediction algorithm. To evaluate dataset size recovery of LoRA weights, we develop and release
> a new benchmark, **LoRA-WISE**, consisting of over 25,000 weight snapshots from more than 2,000 diverse LoRA fine-tuned models. Our best classifier
> can predict the number of fine-tuning images with a mean absolute error of 0.36 images, establishing the feasibility of this attack.

## Task
This paper introduces a new task: dataset size recovery, that aims to determine the number of samples used to train a model, directly from its weights
The setting for the task is as follows:

- The user has access to n different LoRA fine-tuned models, each annotated with its dataset size.

- It is assumed that all n models originated from the same source model and were trained with identical parameters.

- Using only these n observed models, the goal is to predict the dataset size for new models that are trained under the same parameters.

Our method, *DSiRe*, addresses this task, focusing particularly on the important special case of recovering the number of images used to fine-tune a model, 
where fine-tuning was performed via LoRA. DSiRe demonstrates high accuracy in this task, achieving reliable results with just 5 models per dataset size category.

## LoRA-WiSE Benchmark
We present the LoRA Weight Size Evaluation (LoRA-WiSE) benchmark, a comprehensive benchmark specifically designed to evaluate LoRA dataset size recovery methods, for generative models.
- The benchmark can be downloaded from Hugging Face [here](https://huggingface.co/datasets/MoSalama98/LoRA-WiSE).


## Setup
1. Clone this repository
```bash
git clone https://github.com/MoSalama98/DSiRe.git
cd DSiRe
```

2. Create a virtual environment, activate it and install the requirements file:
```bash
python3 -m venv dsire_venv
source dsire_venv/bin/activate
pip install -r requirements.txt
```

## Running DSiRe
The dsire.py script handles the downloading of the LoRA-WiSE dataset that is hosted on Hugging Face.
Below are examples for running DSiRe for dataset size recovery on the LoRA-WiSE benchmark subsets.

#### Low Range 
```bash
python dsire.py --subset="low_32" --rank=32
```
#### Medium Range
```bash
python dsire.py --subset="medium_16" --rank=16
```
#### High Range 
```bash
python dsire.py --subset="high_32" --rank=32
```
## Citation
If you find this useful for your research, please use the following.

**BibTeX:**
```
@article{salama2024dataset,
        title={Dataset Size Recovery from LoRA Weights},
        author={Salama, Mohammad and Kahana, Jonathan and Horwitz, Eliahu and Hoshen, Yedid},
        journal={arXiv preprint arXiv:2406.19395},
        year={2024}
      }
```


## Acknowledgments
- The project makes extensive use of the different Hugging Face libraries (e.g. [Diffusers](https://huggingface.co/docs/diffusers/en/index), [PEFT](https://huggingface.co/docs/peft/en/index), [Transformers](https://huggingface.co/docs/transformers/en/index)).
- The [LoRA-WiSE](https://huggingface.co/datasets/MoSalama98/LoRA-WiSE) benchmark is hosted on Hugging Face.
