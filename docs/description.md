This demo showcases a lightweight model for speech-driven talking-face synthesis, a **28× Compressed Wav2Lip**. The key features of our approach are:
  - compact generator built by removing the residual blocks and reducing the channel width from Wav2Lip.
  - knowledge distillation to effectively train the small-capacity generator without adversarial learning.
  - selective quantization to accelerate inference on edge GPUs without noticeable performance degradation.

<!-- To demonstrate the efficacy of our approach, we provide a latency comparison of different precisions on NVIDIA Jetson edge GPUs in Figure 5. Our approach achieves a remarkable 8× to 17× speedup with FP16 precision, and a 19× speedup on Xavier NX with mixed precision. -->
The below figure shows a latency comparison at different precisions on NVIDIA Jetson edge GPUs, highlighting a 8× to 17× speedup at FP16 and a 19× speedup on Xavier NX at mixed precision.

<center>
    <img alt="compressed-wav2lip-performance" src="https://huggingface.co/spaces/nota-ai/compressed-wav2lip/resolve/2b86e2aa4921d3422f0769ed02dce9898d1e0470/docs/assets/fig5.png" width="70%" />
</center>

<br/>

The generation speed may vary depending on network traffic. Nevertheless, our compresed Wav2Lip _consistently_ delivers a faster inference than the original model, while maintaining similar visual quality. Different from the paper, in this demo, we measure **total processing time** and **FPS** throughout loading the preprocessed video and audio, generating with the model, and merging lip-synced facial images with the original video.

<br/>


### Notice
 - This work was accepted to [Demo] [**ICCV 2023 Demo Track**](https://iccv2023.thecvf.com/demos-111.php); [[Paper](https://arxiv.org/abs/2304.00471)] [**On-Device Intelligence Workshop (ODIW) @ MLSys 2023**](https://sites.google.com/g.harvard.edu/on-device-workshop-23/home); [Poster] [**NVIDIA GPU Technology Conference (GTC) as Poster Spotlight**](https://www.nvidia.com/en-us/on-demand/search/?facet.mimetype[]=event%20session&layout=list&page=1&q=52409&sort=relevance&sortDir=desc).  
 - We thank [NVIDIA Applied Research Accelerator Program](https://www.nvidia.com/en-us/industries/higher-education-research/applied-research-program/) for supporting this research and [Wav2Lip's Authors](https://github.com/Rudrabha/Wav2Lip) for their pioneering research. 