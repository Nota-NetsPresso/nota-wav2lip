import json
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np

import nota_wav2lip.audio as audio
from config import hparams as hp


class VideoSlicer:
    def __init__(self, frame_dir: Union[Path, str], bbox_path: Union[Path, str]):
        self.fps = hp.face.video_fps
        self.frame_dir = frame_dir
        self.frame_path_list = sorted(Path(self.frame_dir).glob("*.jpg"))
        self.frame_array_list: List[np.ndarray] = [cv2.imread(str(image)) for image in self.frame_path_list]

        with open(bbox_path, 'r') as f:
            metadata = json.load(f)
            self.bbox: List[List[int]] = [metadata['bbox'][key] for key in sorted(metadata['bbox'].keys())]
            self.bbox_format = metadata['format']
        assert len(self.bbox) == len(self.frame_array_list)

    def __len__(self):
        return len(self.frame_array_list)

    def __getitem__(self, idx) -> Tuple[np.ndarray, List[int]]:
        bbox = self.bbox[idx]
        frame_original: np.ndarray = self.frame_array_list[idx]
        # return frame_original[bbox[0]:bbox[1], bbox[2]:bbox[3], :]
        return frame_original, bbox


class AudioSlicer:
    def __init__(self, audio_path: Union[Path, str]):
        self.fps = hp.face.video_fps
        self.mel_chunks = self._audio_chunk_generator(audio_path)
        self._audio_path = audio_path

    @property
    def audio_path(self):
        return self._audio_path

    def __len__(self):
        return len(self.mel_chunks)

    def _audio_chunk_generator(self, audio_path):
        wav: np.ndarray = audio.load_wav(audio_path, hp.audio.sample_rate)
        mel: np.ndarray = audio.melspectrogram(wav)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        mel_chunks: List[np.ndarray] = []
        mel_idx_multiplier = 80. / self.fps

        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + hp.face.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - hp.face.mel_step_size:])
                return mel_chunks
            mel_chunks.append(mel[:, start_idx: start_idx + hp.face.mel_step_size])
            i += 1

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.mel_chunks[idx]
