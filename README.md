# Self-Alignment of Large Video Language Models with Refined Regularized Preference Optimization

<a href='https://arxiv.org/abs/2504.12083'><img src='https://img.shields.io/badge/arXiv-paper-red'></a> 
<a href='https://pritamqu.github.io/RRPO/'><img src='https://img.shields.io/badge/project-RRPO-blue'></a> 
<a href='https://huggingface.co/datasets/pritamqu/self-alignment'><img src='https://img.shields.io/badge/huggingface-datasets-green'></a> 
<a href='https://huggingface.co/collections/pritamqu/rrpo-67fbc8c048b298a5fdfb167b'><img src='https://img.shields.io/badge/model-checkpoints-yellow'></a> 
</a><a href='https://github.com/pritamqu/RRPO'><img src='https://img.shields.io/badge/github-repository-purple'></a> 

<h3 align="center">
:star: :star: NeurIPS 2025 :star: :star:
</h3>

Authors: [Pritam Sarkar](https://pritamsarkar.com) and [Ali Etemad](https://www.aiimlab.com/ali-etemad)

This repository provides the official implementation of **[RRPO](https://arxiv.org/abs/2504.12083)**.

<!-- Optionally, add a graphic or model diagram here -->

## Installation

Clone the repository and navigate to the RRPO directory:

```
git clone https://github.com/pritamqu/RRPO
cd RRPO
```

This repository supports three Large Video Language Models (LVLMs), each with its own dependency requirements:

- **[VideoChat2](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2)**: `videochat2.txt`
- **[LLaVA-Video](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/README.md)**: `llavavideo.txt`
- **[LongVU](https://github.com/Vision-CAIR/LongVU)**: `longvu.txt`

#### Example: Setting up LLaVA-Video

Follow similar steps for other models.

```sh
conda create -n llava python=3.10 -y
conda activate llava
pip install -r llavavideo.txt
```

## Models

The self-aligned LVLMs trained with the RRPO loss are released on **Hugging Face**. While these models were trained using **LoRA**, we also provide their **merged weights** to allow for direct use in evaluation and inference tools.

| Model | LoRA | Merged Weights (noqa) |
|------------------------------|------|----------------|
| VideoChat2_stage3_Mistral_7B-RRPO-16f | [pritamqu/VideoChat2_stage3_Mistral_7B-RRPO-16f-LORA](https://huggingface.co/pritamqu/VideoChat2_stage3_Mistral_7B-RRPO-16f-LORA) | - |
| LLaVA-Video-7B-Qwen2-RRPO-16f | [pritamqu/LLaVA-Video-7B-Qwen2-RRPO-16f-LORA](https://huggingface.co/pritamqu/LLaVA-Video-7B-Qwen2-RRPO-16f-LORA) | [pritamqu/LLaVA-Video-7B-Qwen2-RRPO-16f](https://huggingface.co/pritamqu/LLaVA-Video-7B-Qwen2-RRPO-16f) |
| LLaVA-Video-7B-Qwen2-RRPO-32f | [pritamqu/LLaVA-Video-7B-Qwen2-RRPO-32f-LORA](https://huggingface.co/pritamqu/LLaVA-Video-7B-Qwen2-RRPO-32f-LORA) | [pritamqu/LLaVA-Video-7B-Qwen2-RRPO-32f](https://huggingface.co/pritamqu/LLaVA-Video-7B-Qwen2-RRPO-32f) |
| LongVU_Qwen2_7B-RRPO-16f | [pritamqu/LongVU_Qwen2_7B-RRPO-16f-LORA](https://huggingface.co/pritamqu/LongVU_Qwen2_7B-RRPO-16f-LORA) | [pritamqu/LongVU_Qwen2_7B-RRPO-16f](https://huggingface.co/pritamqu/LongVU_Qwen2_7B-RRPO-16f) |

You can download weights as:
```
git clone git@hf.co:pritamqu/LLaVA-Video-7B-Qwen2-RRPO-32f-LORA
```

## Dataset

Our training data is released here [Self-Alignment Dataset](https://huggingface.co/datasets/pritamqu/self-alignment). We release the preferred and non-preferred responses used in self-alignment training. 
```
git clone git@hf.co:datasets/pritamqu/self-alignment
```
The related videos can be downloaded from their original sources. Please check [VideoChat-IT](https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/DATA.md) GitHub page regarding the details of downloading the source videos.

We also share additional details on how to use your own data [here](docs/DATA.md). 

## Training

Before training, make sure to prepare the data and download the weights of the base models. Then you can launch the training jobs as:

VideoChat2
```
bash scripts/videochat2/run.sh
```
LLaVA-Video
```
bash scripts/llavavideo/run.sh
```
LongVU
```
bash scripts/longvu/run.sh
```
The link to the base model weights are:
- [VideoChat2_stage3_Mistral_7B](https://huggingface.co/OpenGVLab/VideoChat2_stage3_Mistral_7B)
- [LLaVA-Video-7B-Qwen2](https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2)
- [LongVU_Qwen2_7B](https://huggingface.co/Vision-CAIR/LongVU_Qwen2_7B)


## Inference

We provide a simple setup to inference using our trained model.

**VideoChat2**
```
bash scripts/inference_videochat2.sh
```

**LLaVA-Video**
```
bash scripts/inference_llavavideo.sh
```

**LongVU**
```
bash scripts/inference_longvu.sh
```

## Results

**RRPO shows consistent improvements over the base model and outperforms DPO across all benchmarks.**

| **Models** | **#F** | **TV Bench** | **Temp Compass** | **Video Hallucer** | **Vid Halluc** | **MV Bench** | **Video MME** | **MLVU** | **LongVideo Bench** |
|------------|------|-------------|----------------|----------------|-------------|-------------|-------------|--------|------------------|
| VideoChat2 | 16 | 44.0 | 59.3 | 23.1 | 73.3 | **60.2** | 41.0 | 46.4 | 40.4 |
| VideoChat2 + DPO | 16 | 45.7 | 60.0 | 22.1 | 72.4 | 59.6 | 43.0 | 47.4 | 41.0 |
| VideoChat2 + **RRPO** | 16 | **45.8** | **60.2** | **32.9** | **76.4** | 59.0 | **44.3** | **47.9** | **42.8** |
|  |  |  |  |  |  |  |  |  |  |
| LLaVA-Video | 64 | 51.0 | 66.0 | 50.0 | 76.6 | 61.1 | 64.0 | 68.6 | 60.1 |
| LLaVA-Video + DPO | 64 | 51.9 | 66.4 | 53.3 | 76.5 | 60.6 | 63.1 | 67.4 | 59.4 |
| LLaVA-Video + **RRPO** | 64 | 51.9 | 66.8 | 55.7 | 76.5 | **62.2** | **64.5** | 69.1 | **60.4** |
| LLaVA-Video + **RRPO** (32f) | 64 | **52.2** | **67.4** | **55.8** | **76.6** | 62.1 | **64.5** | **69.4** | 60.1 |
|  |  |  |  |  |  |  |  |  |  |
| LongVU | 1fps | 53.7 | 63.9 | 39.2 | 67.3 | 65.5 | 56.2 | 63.6 | 48.6 |
| LongVU + DPO | 1fps | 54.3 | 64.3 | 40.9 | 68.5 | 65.9 | 56.6 | 63.6 | 49.4 |
| LongVU + **RRPO** | 1fps | **56.5** | **64.5** | **44.0** | **71.7** | **66.8** | **57.7** | **64.5** | **49.7** |


## Evaluation

You can download evaluation benchmarks from the given links:

- [TVBench](https://huggingface.co/datasets/FunAILab/TVBench)
- [TempCompass](https://huggingface.co/datasets/lmms-lab/TempCompass)
- [VideoHallucer](https://huggingface.co/datasets/bigai-nlco/VideoHallucer)
- [VidHalluc](https://huggingface.co/datasets/chaoyuli/VidHalluc)
- [MVBench](https://huggingface.co/datasets/PKU-Alignment/MVBench)
- [VideoMME](https://huggingface.co/datasets/lmms-lab/Video-MME)
- [MLVU](https://huggingface.co/datasets/MLVU/MVLU)
- [LongVideoBench](https://huggingface.co/datasets/longvideobench/LongVideoBench)

Next, you can run the entire evaluations following the instructions provided [here](./docs/EVALUATION.md).


## Citation

If you find this work useful, please consider citing our paper:

```
@article{sarkar2025rrpo,
  title={Self-alignment of Large Video Language Models with Refined Regularized Preference Optimization},
  author={Sarkar, Pritam and Etemad, Ali},
  journal={arXiv preprint arXiv:2504.12083},
  year={2025}
}
```

## Usage and License Notices

This project incorporates datasets and model checkpoints that are subject to their respective original licenses. Users must adhere to the terms and conditions specified by these licenses.
The assets used in this work include, but are not limited to:
[VideoChat2-IT](https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT),
[VideoChat2_stage3_Mistral_7B](https://huggingface.co/OpenGVLab/VideoChat2_stage3_Mistral_7B),
[LLaVA-Video-7B-Qwen2](https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2),
[LongVU_Qwen2_7B](https://huggingface.co/Vision-CAIR/LongVU_Qwen2_7B). This project does not impose any additional constraints beyond those stipulated in the original licenses. Users must ensure their usage complies with all applicable laws and regulations.
This repository is released under the **Apache 2.0 License**. See [LICENSE](LICENSE) for details.


---
For any issues or questions, please open an issue or contact **Pritam Sarkar** at pritam.sarkar@queensu.ca!






