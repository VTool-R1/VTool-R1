# VTool-R1

This repo contains codes for the paper "VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use"

---


[![Paper](https://img.shields.io/badge/paper-5f16a8?style=for-the-badge&logo=arxiv&logoColor=white&color=FF5F05)](https://arxiv.org/pdf/2505.19255)
[![HOMEPAGE](https://img.shields.io/badge/HOMEPAGE-3858bf?style=for-the-badge&logo=homepage&logoColor=white&color=13294B)](https://vtool-r1.github.io/)
[![Weights](https://img.shields.io/badge/Model%20Weights-63cad3?style=for-the-badge&logo=huggingface&logoColor=white&color=FF5F05)](https://huggingface.co/VTOOL)

# News

- [2025/5/31] Model checkpoints and code available. <!--<span style="color: red;">[**New!**]</span>-->
- [2025/5/25] ArXiv preprint available.


# Introduction

We introduce VTool-R1, the first framework that trains VLMs to generate multimodal chains of thought by interleaving text and intermedi- ate visual reasoning steps. VTool-R1 integrates Python-based visual editing tools into the RFT process, enabling VLMs to learn when and how to generate visual reasoning steps that benefit final reasoning. Trained with outcome-based re- wards tied to task accuracy, our approach elicits strategic visual tool use for reasoning without relying on process-based supervision. Exper- iments on structured visual question answer- ing over charts and tables show that VTool-R1 enhances reasoning performance by teaching VLMs to "think with images" and generate mul- timodal chain of thoughts with tools.

![alt text](vtool_example.png) Figure 2: Qualitative Example from VTool-R1 (3B): The Model Successfully Integrates Intermediate Visual Steps.


## Installation

```
conda create -n vtool python=3.10
conda activate vtool
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install hf_transfer vllm==0.8.3 triton==3.1.0
pip install -e .
```

## Data Preparation (ReFocus)

```
pip install gdown

bash prepare_refocus_data.sh

python chartqa_dataset_creator.py
```

The datasets should be organized into the following structure:

- datasets
  - val_full.parquet
  - train_full.parquet
  - test_full.parquet
  - table_train.parquet
  - table_test.parquet

Alternatively, we provide [preprocessed dataset](https://drive.google.com/drive/folders/16tP_cH-9kGFzjyAn3_z1wo7HQaa9T7MY?usp=share_link) in parquet format.

## Training

Sample training scripts for 3B and 7B are available.

For reference, we trained our 3B models using 8xH100, 7B with 16xH100, both with mixed precision, and 32B with 8 H200 using BF16 precision. You may consider tuning global_batch_size and micro_batch_size_per_device_for_update to reduce VRAM usage. 

Visual tokens for TableQA are often longer, it is recommended to reduce your batch size by half when switching from ChartQA to TableQA.

3B and 7B models require a minimum of 4xH100 and 8xH100 respectively. Training using our specs typically complete within 24 hours.

Due to the nature of our dataset, we used an LLM as our training verifier. Specifically, Qwen/Qwen2.5-7B-Instruct. We recommend either a single H100 or two A100 for stable training (relative to our training specs, e.g. you may find a single A100 or 4090 sufficient for 4xH100 training). You need to first launch the judge using our scripts run_judge.sh. 

If your judge server exists on the same local network as your training machine, then set the environment variable LOCAL_JUDGE=YES. Alternatively, we provide an option to use ngrok if your judge server is on a different network, refer to examples/reward_function/refocus_llm.py to configure your ngrok domain.

## Questions

Please open an issue if your have any questions.

## Model Use and Evaluation

We provide 3B, 7B, and 32B model weights on ChartQA and TableQA datasets from ReFocus. Download them from [Hugging Face](https://huggingface.co/VTOOL).

Evaluation scripts are in the eval folder. For evaluation, we use ChatGPT from OpenAI as the judge. Please configure your own API keys.

## Acknowledgement

This research used the Delta advanced computing and data resource which is supported by the National Science Foundation (award OAC 2005572) and the State of Illinois. Delta is a joint effort of the University of Illinois Urbana-Champaign and its National Center for Supercomputing Applications.

We would also like to acknowledge Bowen Jin (author of Search-R1) and Xingyu Fu (author of Refocus) for their valuable suggestions and contributions to our project.

We also thank [veRL](https://github.com/volcengine/verl) and [EasyR1](https://github.com/hiyouga/EasyR1) for providing the essential VLM RL infrastructure.

This work was supported by the National Science Foundation grants NSF CNS 21-06592, NSF OAC 18-35834 KN, NSF CNS 19-00875 and NSF CCF 22-17144. Any results and opinions are our own and do not represent views of National Science Foundation.

## BibTex

If you find our project helpful, please cite:

<pre style="background-color: auto; padding: 0.8rem 1rem 0.4rem 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.9rem;">
@misc{wu2025vtoolr1vlmslearnthink,
      title={VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use}, 
      author={Mingyuan Wu and Jingcheng Yang and Jize Jiang and Meitang Li and Kaizhuo Yan and Hanchao Yu and Minjia Zhang and Chengxiang Zhai and Klara Nahrstedt},
      year={2025},
      eprint={2505.19255},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.19255}, 
}
</pre>