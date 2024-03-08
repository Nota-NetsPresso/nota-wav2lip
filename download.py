import argparse

from nota_wav2lip.preprocess import get_cropped_face_from_lrs3_label


def parse_args():

    parser = argparse.ArgumentParser(description="NotaWav2Lip: Get LRS3 video sample with the label text file")

    parser.add_argument(
        '-i',
        '--input-file',
        type=str,
        required=True,
        help="Path of the label text file downloaded from https://mmai.io/datasets/lip_reading"
    )

    parser.add_argument(
        '-o',
        '--output-dir',
        type=str,
        default="sample_video_lrs3",
        help="Output directory to save the result. Defaults: sample_video_lrs3"
    )

    parser.add_argument(
        '--ignore-cache',
        action='store_true',
        help="Whether to force downloading and resampling video and overwrite pre-existing files"
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    get_cropped_face_from_lrs3_label(
        args.input_file,
        video_root_dir=args.output_dir,
        ignore_cache = args.ignore_cache
    )
