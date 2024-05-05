<div align="center">

# Discharge LLM

</div>

This repository maintains the code, data, and model checkpoints for the paper *Chain-of-Thought (CoT) Instruction Finetuning Large Language Models For Discharge Summary Documentation*
It is a part of our approach in the [Discharge Me!](https://www.codabench.org/competitions/2008/) shared task collocated collocated with the 23th Workshop on Biomedical Natural Language Processing (BioNLP).

## Installation
It is recommended to set up the environment and install required libraries using conda. 
It is also recommended that the machine should have GPUs to perform inference at a reasonable time.  
### 1. Create new virtual environment by
```bash
conda create --name pakpa python=3.9
conda activate pakpa
```
### 2. Install Pytorch
#### Windows or Linux
##### Using GPUs
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
##### Using CPU
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
#### Mac
```bash
conda install pytorch::pytorch torchvision torchaudio -c pytorch
```
For other versions, please visit: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

# More code to release soon.