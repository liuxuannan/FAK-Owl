# [FKA-Owl: Advancing Multimodal Fake News Detection through Knowledge-Augmented LVLMs (ACM MM 2024)](https://arxiv.org/abs/2403.01988)

[![Project page](https://img.shields.io/badge/Project-Page-brightgreen)](https://liuxuannan.github.io/FKA_Owl.github.io/)
[![paper](https://img.shields.io/badge/Paper-ACMMM-brightgreen)](https://arxiv.org/abs/2403.01988)
[![arXiv](https://img.shields.io/badge/ArXiv-2403.01988-brightgreen)](https://arxiv.org/abs/2403.01988)

---

[//]: # (https://user-images.githubusercontent.com/54032224/302051504-dac634f3-85ef-4ff1-80a2-bd2805e067ea.mp4)

### Introduction: 

<p align="center" width="100%">
<img src="./images/forgery-knowledge.jpg" alt="FKA_Owl_logo" style="width: 80%; min-width: 400px; display: block; margin: auto;" />
</p>

**FKA-Owl** pioneers leveraging rich world knowledge from large vision-language models (LVLMs) and enhancing them with forgery-specific knowledge, to tackle the domain shift issue in multimodal fake news detection. We propose two lightweight modules for forgery-specific knowledge augmentation: the cross-modal reasoning module and the visual-artifact localization module to extract semantic correlations and artifact traces, respectively.

<img src="./images/framework.jpg" alt="FKA_Owl" style="zoom:100%;" />
The proposed FKA-Owl is built upon the off-the-shelf LVLM consisting of an image encoder and a Large Language Model (LLM). Given a manipulated image-text pair, the cross-modal reasoning module (a) first extracts cross-modal semantic embeddings and visual patch features. Then, these visual patch features are processed by the visual-artifact localization module (b) to encode precise artifact embeddings. Finally, the semantic and artifact embeddings are incorporated into the forgery-aware vision-language model (c) combined with image features and the human prompt for deep manipulation reasoning.


# 🔧 Dependencies and Installation

- Python = 3.9.0
- [PyTorch= 1.13.1, torchvision=0.14.1 ]( https://pytorch.org/get-started/previous-versions/) 

```bash
# create an environment
conda create -n FKA_Owl python==3.9.0
# activate the environment
conda activate FKA_Owl
# install pytorch using pip
# for example: for Linux with CUDA 11.7
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# install other dependencies
pip install -r requirements.txt
```


# ⏬ Prepare Checkpoint 

You can download the pre-trained ImageBind model using [this link](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth). After downloading, put the downloaded file (imagebind_huge.pth) in [[./pretrained_ckpt/imagebind_ckpt/]](./pretrained_ckpt/imagebind_ckpt/) directory. 

To prepare the pre-trained Vicuna model, please follow the instructions provided [[here]](./pretrained_ckpt#1-prepare-vicuna-checkpoint).  

We use the pre-trained parameters from [PandaGPT](https://github.com/yxuansu/PandaGPT) to initialize our model. You can get the weights of PandaGPT trained with different strategies in the table below. In our experiments, we use the Vicuna-7B and `openllmplayground/pandagpt_7b_max_len_1024` due to the limitation of computation resource. Please put the downloaded 7B delta weights file (pytorch_model.pt) in the [./pretrained_ckpt/pandagpt_ckpt/7b/](./pretrained_ckpt/pandagpt_ckpt/7b/).


# ⏬ Prepare Data
 You can download DGM4 dataset from [[this link]](https://github.com/rshaojimmy/MultiModal-DeepFake). After downloading, put the data in the [[./data]](./data/) directory.

 The directory of [[./data]](./data/) should look like:

 ```
└── data
    └── DGM4
        ├── manipulation
        │   ├── infoswap
        │   |   ├── ...
        |   |   └── xxxxxx.jpg
        │   ├── simswap
        │   |   ├── ...
        |   |   └── xxxxxx.jpg
        │   ├── StyleCLIP
        │   |   ├── ...
        |   |   └── xxxxxx.jpg
        │   └── HFGI
        │       ├── ...
        |       └── xxxxxx.jpg
        ├── origin
        │   ├── gardian
        │   |   ├── ...
        |   |   ...
        |   |   └── xxxx
        │   |       ├── ...
        │   |       ...
        │   |       └── xxxxxx.jpg
        │   ├── usa_today
        │   |   ├── ...
        |   |   ...
        |   |   └── xxxx
        │   |       ├── ...
        │   |       ...
        │   |       └── xxxxxx.jpg
        │   ├── washington_post
        │   |   ├── ...
        |   |   ...
        |   |   └── xxxx
        │   |       ├── ...
        │   |       ...
        │   |       └── xxxxxx.jpg
        │   └── bbc
        │       ├── ...
        |       ...
        |       └── xxxx
        │           ├── ...
        │           ...
        │           └── xxxxxx.jpg
        └── metadata_split
            ├── bbc
            |    ├── train.json
            |    ├── test.json
            |    └── val.json
            ├── guardian
            |    ├── train.json
            |    ├── test.json
            |    └── val.json
            ├── usa_today
            |    ├── train.json
            |    ├── test.json
            |    └── val.json
            ├── washington_post
            |    ├── train.json
            |    ├── test.json
            |    └── val.json
```

# 💻 Training FKA-Owl

To train FAK-Owl on the bbc subset of DGM4 dataset, please run the following commands:
```yaml
cd ./code
bash ./scripts/train_DGM4_bbc.sh
```

The key arguments of the training script are as follows:
* `--config_path`: The data path for the config file `train_bbc.yaml`.
* `--imagebind_ckpt_path`: The path of ImageBind checkpoint.
* `--vicuna_ckpt_path`: The directory that saves the pre-trained Vicuna checkpoints.
* `--max_tgt_len`: The maximum sequence length of training instances.
* `--save_path`: The directory which saves the trained delta weights. This directory will be automatically created.
* `--log_path`: The directory which saves the log. This directory will be automatically created.

Note that the epoch number can be set in the `epochs` argument at [./code/config/openllama_peft.yaml](./code/config/openllama_peft.yaml) file and the learning rate can be set in  [./code/dsconfig/openllama_peft_stage_1.json](./code/dsconfig/openllama_peft_stage_1.json)

# 💻 Testing FKA-Owl

To testing FAK-Owl on the washington_post subset of DGM4 dataset, please run the following commands:
```yaml
cd ./code
bash test.sh
```

# 🤗 Acknowledgements
We borrow some codes and the pre-trained weights from [PandaGPT](https://github.com/yxuansu/PandaGPT). Thanks for their wonderful work!


# Citation:
If you found FKA-Owl useful in your research or applications, please kindly cite using the following BibTeX:
```
@inproceedings{liu2024fka,
    title={FKA-Owl: Advancing Multimodal Fake News Detection through Knowledge-Augmented LVLMs},
    author={Liu, Xuannan and Li, Peipei and Huang, Huaibo and Li, Zekun and Cui, Xing and Liang, Jiahao and Qin, Lixiong and Deng, Weihong and He, Zhaofeng},
    booktitle={ACM MM},
    year={2024}
}
```




















