---
title: Compressed Wav2Lip
emoji: 🌟
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 4.13.0
app_file: app.py
pinned: true
license: apache-2.0
---

# 28× Compressed Wav2Lip by Nota AI

Official codebase for [**Accelerating Speech-Driven Talking Face Generation with 28× Compressed Wav2Lip**](https://arxiv.org/abs/2304.00471).

- Presented at [ICCV'23 Demo](https://iccv2023.thecvf.com/demos-111.php) Track; [On-Device Intelligence Workshop](https://sites.google.com/g.harvard.edu/on-device-workshop-23/home) @ MLSys'23; [NVIDIA GTC 2023](https://www.nvidia.com/en-us/on-demand/search/?facet.mimetype[]=event%20session&layout=list&page=1&q=52409&sort=relevance&sortDir=desc) Poster.


## Installation
#### Docker (recommended)
```bash
git clone https://github.com/Nota-NetsPresso/nota-wav2lip.git
cd nota-wav2lip
docker compose run --service-ports --name nota-compressed-wav2lip compressed-wav2lip bash
```

#### Conda
<details>
<summary>Click</summary>

```bash
git clone https://github.com/Nota-NetsPresso/nota-wav2lip.git
cd nota-wav2lip
apt-get update
apt-get install ffmpeg libsm6 libxext6 tmux git -y
conda create -n nota-wav2lip python=3.9
conda activate nota-wav2lip
pip install -r requirements.txt
```
</details>

## Gradio Demo
Use the below script to run the [nota-ai/compressed-wav2lip demo](https://huggingface.co/spaces/nota-ai/compressed-wav2lip). The models and sample data will be downloaded automatically.

  ```bash
  bash app.sh
  ```

## Inference
(1) Download YouTube videos in the LRS3-TED label text file and preprocess them properly.
  - Download `lrs3_v0.4_txt.zip` from [this link](https://mmai.io/datasets/lip_reading/).
  - Unzip the file and make a folder structure: `./data/lrs3_v0.4_txt/lrs3_v0.4/test`
  - Run `bash download.sh`
  - Run `bash preprocess.sh`

(2) Run the script to compare the original Wav2Lip with Nota's compressed version.

  ```bash
  bash inference.sh
  ```

## License
- All rights related to this repository and the compressed models are reserved by Nota Inc.
- The intended use is strictly limited to research and non-commercial projects.

## Contact
- To obtain compression code and assistance, kindly contact Nota AI (contact@nota.ai). These are provided as part of our business solutions.
- For Q&A about this repo, use this board: [Nota-NetsPresso/discussions](https://github.com/orgs/Nota-NetsPresso/discussions)

## Acknowledgment
 - [NVIDIA Applied Research Accelerator Program](https://www.nvidia.com/en-us/industries/higher-education-research/applied-research-program/) for supporting this research.
 - [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) and [LRS3-TED](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/) for facilitating the development of the original Wav2Lip.

## Citation
```bibtex
@article{kim2023unified,
      title={A Unified Compression Framework for Efficient Speech-Driven Talking-Face Generation}, 
      author={Kim, Bo-Kyeong and Kang, Jaemin and Seo, Daeun and Park, Hancheol and Choi, Shinkook and Song, Hyoung-Kyu and Kim, Hyungshin and Lim, Sungsu},
      journal={MLSys Workshop on On-Device Intelligence (ODIW)},
      year={2023},
      url={https://arxiv.org/abs/2304.00471}
}
```