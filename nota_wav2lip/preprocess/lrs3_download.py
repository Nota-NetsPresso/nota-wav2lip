import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict, Union

import cv2
import numpy as np
import yt_dlp
from loguru import logger
from tqdm import tqdm

from nota_wav2lip.util import FFMPEG_LOGGING_MODE


class LabelInfo(TypedDict):
    text: str
    conf: int
    url: str
    bbox_xywhn: Dict[int, Tuple[float, float, float, float]]

def frame_to_time(frame_id: int, fps=25) -> str:
    seconds = frame_id / fps

    hours = int(seconds // 3600)
    seconds -= 3600 * hours

    minutes = int(seconds // 60)
    seconds -= 60 * minutes

    seconds_int = int(seconds)
    seconds_milli = int((seconds - int(seconds)) * 1e3)

    return f"{hours:02d}:{minutes:02d}:{seconds_int:02d}.{seconds_milli:03d}"  # HH:MM:SS.mmm

def save_audio_file(input_path, start_frame_id, to_frame_id, output_path=None):
    input_path = Path(input_path)
    output_path = output_path if output_path is not None else input_path.with_suffix('.wav')

    ss = frame_to_time(start_frame_id)
    to = frame_to_time(to_frame_id)
    subprocess.call(
        f"ffmpeg {FFMPEG_LOGGING_MODE['ERROR']} -y -i {input_path} -vn -acodec pcm_s16le -ss {ss} -to {to} -ar 16000 -ac 1 {output_path}",
        shell=platform.system() != 'Windows'
    )

def merge_video_audio(video_path, audio_path, output_path):
    subprocess.call(
        f"ffmpeg {FFMPEG_LOGGING_MODE['ERROR']} -y -i {video_path} -i {audio_path} -strict experimental {output_path}",
        shell=platform.system() != 'Windows'
    )

def parse_lrs3_label(label_path) -> LabelInfo:
    label_text = Path(label_path).read_text()
    label_splitted = label_text.split('\n')

    # Label validation
    assert label_splitted[0].startswith("Text:")
    assert label_splitted[1].startswith("Conf:")
    assert label_splitted[2].startswith("Ref:")
    assert label_splitted[4].startswith("FRAME")

    label_info = LabelInfo(bbox_xywhn={})
    label_info['text'] = label_splitted[0][len("Text:  "):].strip()
    label_info['conf'] = int(label_splitted[1][len("Conf:  "):])
    label_info['url'] = label_splitted[2][len("Ref:  "):].strip()

    for label_line in label_splitted[5:]:
        bbox_splitted = [x.strip() for x in label_line.split('\t')]
        if len(bbox_splitted) != 5:
            continue
        frame_index = int(bbox_splitted[0])
        bbox_xywhn = tuple(map(float, bbox_splitted[1:]))
        label_info['bbox_xywhn'][frame_index] = bbox_xywhn

    return label_info

def _get_cropped_bbox(bbox_info_xywhn, original_width, original_height):

    bbox_info = bbox_info_xywhn
    x = bbox_info[0] * original_width
    y = bbox_info[1] * original_height
    w = bbox_info[2] * original_width
    h = bbox_info[3] * original_height

    x_min = max(0, int(x - 0.5 * w))
    y_min = max(0, int(y))
    x_max = min(original_width, int(x + 1.5 * w))
    y_max = min(original_height, int(y + 1.5 * h))

    cropped_width = x_max - x_min
    cropped_height = y_max - y_min

    if cropped_height > cropped_width:
        offset = cropped_height - cropped_width
        offset_low = min(x_min, offset // 2)
        offset_high = min(offset - offset_low, original_width - x_max)
        x_min -= offset_low
        x_max += offset_high
    else:
        offset = cropped_width - cropped_height
        offset_low = min(y_min, offset // 2)
        offset_high = min(offset - offset_low, original_width - y_max)
        y_min -= offset_low
        y_max += offset_high

    return x_min, y_min, x_max, y_max

def _get_smoothened_boxes(bbox_dict, bbox_smoothen_window):
    boxes = [np.array(bbox_dict[frame_id]) for frame_id in sorted(bbox_dict)]
    for i in range(len(boxes)):
        window = boxes[len(boxes) - bbox_smoothen_window:] if i + bbox_smoothen_window > len(boxes) else boxes[i:i + bbox_smoothen_window]
        boxes[i] = np.mean(window, axis=0)

    for idx, frame_id in enumerate(sorted(bbox_dict)):
        bbox_dict[frame_id] = (np.rint(boxes[idx])).astype(int).tolist()
    return bbox_dict

def download_video_from_youtube(youtube_ref, output_path):
    ydl_url = f"https://www.youtube.com/watch?v={youtube_ref}"
    ydl_opts = {
        'format': 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]',
        'outtmpl': str(output_path),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([ydl_url])

def resample_video(input_path, output_path):
    subprocess.call(
        f"ffmpeg {FFMPEG_LOGGING_MODE['INFO']} -y -i {input_path} -r 25 -preset veryfast {output_path}",
        shell=platform.system() != 'Windows'
    )

def _get_smoothen_xyxy_bbox(
    label_bbox_xywhn: Dict[int, Tuple[float, float, float, float]],
    original_width: int,
    original_height: int,
    bbox_smoothen_window: int = 5
) -> Dict[int, Tuple[float, float, float, float]]:

    label_bbox_xyxy: Dict[int, Tuple[float, float, float, float]] = {}
    for frame_id in sorted(label_bbox_xywhn):
        frame_bbox_xywhn = label_bbox_xywhn[frame_id]
        bbox_xyxy = _get_cropped_bbox(frame_bbox_xywhn, original_width, original_height)
        label_bbox_xyxy[frame_id] = bbox_xyxy

    label_bbox_xyxy = _get_smoothened_boxes(label_bbox_xyxy, bbox_smoothen_window=bbox_smoothen_window)
    return label_bbox_xyxy

def get_start_end_frame_id(
    label_bbox_xywhn: Dict[int, Tuple[float, float, float, float]],
) -> Tuple[int, int]:
    frame_ids = list(label_bbox_xywhn.keys())
    start_frame_id = min(frame_ids)
    to_frame_id = max(frame_ids)
    return start_frame_id, to_frame_id

def crop_video_with_bbox(
    input_path,
    label_bbox_xywhn: Dict[int, Tuple[float, float, float, float]],
    start_frame_id,
    to_frame_id,
    output_path,
    bbox_smoothen_window = 5,
    frame_width = 224,
    frame_height = 224,
    fps = 25,
    interpolation = cv2.INTER_CUBIC,
):
    def frame_generator(cap):
        if not cap.isOpened():
            raise IOError("Error: Could not open video.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame

        cap.release()

    cap = cv2.VideoCapture(str(input_path))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    label_bbox_xyxy = _get_smoothen_xyxy_bbox(label_bbox_xywhn, original_width, original_height, bbox_smoothen_window=bbox_smoothen_window)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    for frame_id, frame in tqdm(enumerate(frame_generator(cap))):
        if start_frame_id <= frame_id <= to_frame_id:
            x_min, y_min, x_max, y_max = label_bbox_xyxy[frame_id]

            frame_cropped = frame[y_min:y_max, x_min:x_max]
            frame_cropped = cv2.resize(frame_cropped, (frame_width, frame_height), interpolation=interpolation)
            out.write(frame_cropped)

    out.release()


def get_cropped_face_from_lrs3_label(
    label_text_path: Union[Path, str],
    video_root_dir: Union[Path, str],
    bbox_smoothen_window: int = 5,
    frame_width: int = 224,
    frame_height: int = 224,
    fps: int = 25,
    interpolation = cv2.INTER_CUBIC,
    ignore_cache: bool = False,
):
    label_text_path = Path(label_text_path)
    label_info = parse_lrs3_label(label_text_path)
    start_frame_id, to_frame_id = get_start_end_frame_id(label_info['bbox_xywhn'])

    video_root_dir = Path(video_root_dir)
    video_cache_dir = video_root_dir / ".cache"
    video_cache_dir.mkdir(parents=True, exist_ok=True)

    output_video: Path = video_cache_dir / f"{label_info['url']}.mp4"
    output_resampled_video: Path = output_video.with_name(f"{output_video.stem}-25fps.mp4")
    output_cropped_audio: Path = output_video.with_name(f"{output_video.stem}-{label_text_path.stem}-cropped.wav")
    output_cropped_video: Path = output_video.with_name(f"{output_video.stem}-{label_text_path.stem}-cropped.mp4")
    output_cropped_with_audio: Path = video_root_dir / output_video.with_name(f"{output_video.stem}-{label_text_path.stem}.mp4").name

    if not output_video.exists() or ignore_cache:
        youtube_ref = label_info['url']
        logger.info(f"Download Youtube video(https://www.youtube.com/watch?v={youtube_ref}) ... will be saved at {output_video}")
        download_video_from_youtube(youtube_ref, output_path=output_video)

    if not output_resampled_video.exists() or ignore_cache:
        logger.info(f"Resampling video to 25 FPS ... will be saved at {output_resampled_video}")
        resample_video(input_path=output_video, output_path=output_resampled_video)

    if not output_cropped_audio.exists() or ignore_cache:
        logger.info(f"Cut audio file with the given timestamps ... will be saved at {output_cropped_audio}")
        save_audio_file(
            output_resampled_video,
            start_frame_id=start_frame_id,
            to_frame_id=to_frame_id,
            output_path=output_cropped_audio
        )

    logger.info(f"Naive crop the face region with the given frame labels ... will be saved at {output_cropped_video}")
    crop_video_with_bbox(
        output_resampled_video,
        label_info['bbox_xywhn'],
        start_frame_id,
        to_frame_id,
        output_path=output_cropped_video,
        bbox_smoothen_window=bbox_smoothen_window,
        frame_width=frame_width,
        frame_height=frame_height,
        fps=fps,
        interpolation=interpolation
    )

    if not output_cropped_with_audio.exists() or ignore_cache:
        logger.info(f"Merge an audio track with the cropped face sequence ... will be saved at {output_cropped_with_audio}")
        merge_video_audio(output_cropped_video, output_cropped_audio, output_cropped_with_audio)
