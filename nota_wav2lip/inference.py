from typing import Iterable, Iterator, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from tqdm import tqdm

from config import hparams as hp
from nota_wav2lip.models.util import count_params, load_model


class Wav2LipInferenceImpl:
    def __init__(self, model_name: str, hp_inference_model: DictConfig, device='cpu'):
        self.model: nn.Module = load_model(
            model_name,
            device=device,
            **hp_inference_model
        )
        self.device = device
        self._params: str = self._format_param(count_params(self.model))

    @property
    def params(self):
        return self._params

    @staticmethod
    def _format_param(num_params: int) -> str:
        params_in_million = num_params / 1e6
        return f"{params_in_million:.1f}M"

    @staticmethod
    def _reset_batch() -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[List[int]]]:
        return [], [], [], []

    def get_data_iterator(
        self,
        audio_iterable: Iterable[np.ndarray],
        video_iterable: List[Tuple[np.ndarray, List[int]]]
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]]:
        img_batch, mel_batch, frame_batch, coords_batch = self._reset_batch()

        for i, m in enumerate(audio_iterable):
            idx = i % len(video_iterable)
            _frame_to_save, coords = video_iterable[idx]
            frame_to_save = _frame_to_save.copy()
            face = frame_to_save[coords[0]:coords[1], coords[2]:coords[3]].copy()

            face: np.ndarray = cv2.resize(face, (hp.face.img_size, hp.face.img_size))

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= hp.inference.batch_size:
                img_batch = np.asarray(img_batch)
                mel_batch = np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, hp.face.img_size // 2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = self._reset_batch()

        if len(img_batch) > 0:
            img_batch = np.asarray(img_batch)
            mel_batch = np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, hp.face.img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch

    @torch.no_grad()
    def inference_with_iterator(
        self,
        audio_iterable: Iterable[np.ndarray],
        video_iterable: List[Tuple[np.ndarray, List[int]]]
    ) -> Iterator[np.ndarray]:
        data_iterator = self.get_data_iterator(audio_iterable, video_iterable)

        for (img_batch, mel_batch, frames, coords) in \
            tqdm(data_iterator, total=int(np.ceil(float(len(audio_iterable)) / hp.inference.batch_size))):

            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)

            preds: torch.Tensor = self.forward(mel_batch, img_batch)

            preds = preds.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            for pred, frame, coord in zip(preds, frames, coords):
                y1, y2, x1, x2 = coord
                pred = cv2.resize(pred.astype(np.uint8), (x2 - x1, y2 - y1))

                frame[y1:y2, x1:x2] = pred
                yield frame

    @torch.no_grad()
    def forward(self, audio_sequences: torch.Tensor, face_sequences: torch.Tensor) -> torch.Tensor:
        return self.model(audio_sequences, face_sequences)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
