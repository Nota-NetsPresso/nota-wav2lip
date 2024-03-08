import json
import platform
import subprocess
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

import face_detection
from nota_wav2lip.util import FFMPEG_LOGGING_MODE

detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device='cpu')
PADDING = [0, 10, 0, 0]


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        window = boxes[len(boxes) - T:] if i + T > len(boxes) else boxes[i:i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def face_detect(images, pads, no_smooth=False, batch_size=1):

    predictions = []
    images_array = [cv2.imread(str(image)) for image in images]
    for i in tqdm(range(0, len(images_array), batch_size)):
        predictions.extend(detector.get_detections_for_batch(np.array(images_array[i:i + batch_size])))

    results = []
    pady1, pady2, padx1, padx2 = pads
    for rect, image_array in zip(predictions, images_array):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image_array)  # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image_array.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image_array.shape[1], rect[2] + padx2)
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    bbox_format = "(y1, y2, x1, x2)"
    if not no_smooth:
        boxes = get_smoothened_boxes(boxes, T=5)
    outputs = {
        'bbox': {str(image_path): tuple(map(int, (y1, y2, x1, x2))) for image_path, (x1, y1, x2, y2) in zip(images, boxes)},
        'format': bbox_format
    }
    return outputs


def save_video_frame(video_path, output_dir=None):
    video_path = Path(video_path)
    output_dir = output_dir if output_dir is not None else video_path.with_suffix('')
    output_dir.mkdir(exist_ok=True)
    return subprocess.call(
        f"ffmpeg {FFMPEG_LOGGING_MODE['ERROR']} -y -i {video_path} -r 25 -f image2 {output_dir}/%05d.jpg",
        shell=platform.system() != 'Windows'
    )


def save_audio_file(video_path, output_path=None):
    video_path = Path(video_path)
    output_path = output_path if output_path is not None else video_path.with_suffix('.wav')
    subprocess.call(
        f"ffmpeg {FFMPEG_LOGGING_MODE['ERROR']} -y -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {output_path}",
        shell=platform.system() != 'Windows'
    )


def save_bbox_file(video_path, bbox_dict, output_path=None):
    video_path = Path(video_path)
    output_path = output_path if output_path is not None else video_path.with_suffix('.json')

    with open(output_path, 'w') as f:
        json.dump(bbox_dict, f, indent=4)

def get_preprocessed_data(video_path: Path):
    video_path = Path(video_path)

    image_sequence_dir = video_path.with_suffix('')
    audio_path = video_path.with_suffix('.wav')
    face_bbox_json_path = video_path.with_suffix('.json')

    logger.info(f"Save 25 FPS video frames as image files ... will be saved at {video_path}")
    save_video_frame(video_path=video_path, output_dir=image_sequence_dir)

    logger.info(f"Save the audio as wav file ... will be saved at {audio_path}")
    save_audio_file(video_path=video_path, output_path=audio_path)  # bonus

    # Load images, extract bboxes and save the coords(to directly use as array indicies)
    logger.info(f"Extract face boxes and save the coords with json format ... will be saved at {face_bbox_json_path}")
    results = face_detect(sorted(image_sequence_dir.glob("*.jpg")), pads=PADDING)
    save_bbox_file(video_path, results, output_path=face_bbox_json_path)
