from omegaconf import DictConfig, OmegaConf

hparams: DictConfig = OmegaConf.load("config/nota_wav2lip.yaml")

hparams_gradio: DictConfig = OmegaConf.load("config/gradio.yaml")
