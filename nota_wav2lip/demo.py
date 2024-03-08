import argparse
import platform
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Union

import cv2
import numpy as np

from config import hparams as hp
from nota_wav2lip.inference import Wav2LipInferenceImpl
from nota_wav2lip.util import FFMPEG_LOGGING_MODE
from nota_wav2lip.video import AudioSlicer, VideoSlicer


class Wav2LipModelComparisonDemo:
    def __init__(self, device='cpu', result_dir='./temp', model_list: Optional[Union[str, List[str]]]=None):
        if model_list is None:
            model_list: List[str] = ['wav2lip', 'nota_wav2lip']
        if isinstance(model_list, str) and len(model_list) != 0:
            model_list: List[str] = [model_list]
        super().__init__()
        self.video_dict: Dict[str, VideoSlicer] = {}
        self.audio_dict: Dict[str, AudioSlicer] = {}

        self.model_zoo: Dict[str, Wav2LipInferenceImpl] = {}
        for model_name in model_list:
            assert model_name in hp.inference.model, f"{model_name} not in hp.inference_model: {hp.inference.model}"
            self.model_zoo[model_name] = Wav2LipInferenceImpl(
                model_name, hp_inference_model=hp.inference.model[model_name], device=device
            )

        self._params_zoo: Dict[str, str] = {
            model_name: self.model_zoo[model_name].params for model_name in self.model_zoo
        }

        self.result_dir: Path = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)

    @property
    def params(self):
        return self._params_zoo

    def _infer(
        self,
        audio_name: str,
        video_name: str,
        model_type: Literal['wav2lip', 'nota_wav2lip']
    ) -> Iterator[np.ndarray]:
        audio_iterable: AudioSlicer = self.audio_dict[audio_name]
        video_iterable: VideoSlicer = self.video_dict[video_name]
        target_model = self.model_zoo[model_type]
        return target_model.inference_with_iterator(audio_iterable, video_iterable)

    def update_audio(self, audio_path, name=None):
        _name = name if name is not None else Path(audio_path).stem
        self.audio_dict.update(
            {_name: AudioSlicer(audio_path)}
        )

    def update_video(self, frame_dir_path, bbox_path, name=None):
        _name = name if name is not None else Path(frame_dir_path).stem
        self.video_dict.update(
            {_name: VideoSlicer(frame_dir_path, bbox_path)}
        )

    def save_as_video(self, audio_name, video_name, model_type):

        output_video_path = self.result_dir / 'generated_with_audio.mp4'
        frame_only_video_path = self.result_dir / 'generated.mp4'
        audio_path = self.audio_dict[audio_name].audio_path

        out = cv2.VideoWriter(str(frame_only_video_path),
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              hp.face.video_fps,
                              (hp.inference.frame.w, hp.inference.frame.h))
        start = time.time()
        for frame in self._infer(audio_name=audio_name, video_name=video_name, model_type=model_type):
            out.write(frame)
        inference_time = time.time() - start
        out.release()

        command = f"ffmpeg {FFMPEG_LOGGING_MODE['ERROR']} -y -i {audio_path} -i {frame_only_video_path} -strict -2 -q:v 1 {output_video_path}"
        subprocess.call(command, shell=platform.system() != 'Windows')

        # The number of frames of generated video
        video_frames_num = len(self.audio_dict[audio_name])
        inference_fps = video_frames_num / inference_time

        return output_video_path, inference_time, inference_fps
