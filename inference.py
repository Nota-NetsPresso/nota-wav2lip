import argparse
import os
import subprocess
from pathlib import Path

from config import hparams as hp
from nota_wav2lip import Wav2LipModelComparisonDemo

LRS_ORIGINAL_URL = os.getenv('LRS_ORIGINAL_URL', None)
LRS_COMPRESSED_URL = os.getenv('LRS_COMPRESSED_URL', None)

if not Path(hp.inference.model.wav2lip.checkpoint).exists() and LRS_ORIGINAL_URL is not None:
    subprocess.call(f"wget --no-check-certificate -O {hp.inference.model.wav2lip.checkpoint} {LRS_ORIGINAL_URL}", shell=True)
if not Path(hp.inference.model.nota_wav2lip.checkpoint).exists() and LRS_COMPRESSED_URL is not None:
    subprocess.call(f"wget --no-check-certificate -O {hp.inference.model.nota_wav2lip.checkpoint} {LRS_COMPRESSED_URL}", shell=True)

def parse_args():

    parser = argparse.ArgumentParser(description="NotaWav2Lip: Inference snippet for your own video and audio pair")

    parser.add_argument(
        '-a',
        '--audio-input',
        type=str,
        required=True,
        help="Path of the audio file"
    )

    parser.add_argument(
        '-v',
        '--video-frame-input',
        type=str,
        required=True,
        help="Input directory with face image sequence. We recommend to extract the face image sequence with `preprocess.py`."
    )

    parser.add_argument(
        '-b',
        '--bbox-input',
        type=str,
        help="Path of the file with bbox coordinates. We recommend to extract the json file with `preprocess.py`."
             "If None, it pretends that the json file is located at the same directory with face images: {VIDEO_FRAME_INPUT}.with_suffix('.json')."
    )

    parser.add_argument(
        '-m',
        '--model',
        choices=['wav2lip', 'nota_wav2lip'],
        default='nota_wav2ilp',
        help="Model for generating talking video. Defaults: nota_wav2lip"
    )

    parser.add_argument(
        '-o',
        '--output-dir',
        type=str,
        default="result",
        help="Output directory to save the result. Defaults: result"
    )

    parser.add_argument(
        '-d',
        '--device',
        choices=['cpu', 'cuda'],
        default='cpu',
        help="Device setting for model inference. Defaults: cpu"
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    bbox_input = args.bbox_input if args.bbox_input is not None \
        else Path(args.video_frame_input).with_suffix('.json')

    servicer = Wav2LipModelComparisonDemo(device=args.device, result_dir=args.output_dir, model_list=args.model)
    servicer.update_audio(args.audio_input, name='a0')
    servicer.update_video(args.video_frame_input, bbox_input, name='v0')

    servicer.save_as_video('a0', 'v0', args.model)
