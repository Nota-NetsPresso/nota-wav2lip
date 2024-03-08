import argparse

from nota_wav2lip.preprocess import get_preprocessed_data


def parse_args():

    parser = argparse.ArgumentParser(description="NotaWav2Lip: Preprocess the facial video with face detection")

    parser.add_argument(
        '-i',
        '--input-file',
        type=str,
        required=True,
        help="Path of the facial video. We recommend that the video is one of LRS3 data samples, which is the result of `download.py`."
             "The extracted features and facial image sequences are saved at the same location with the input file."
    )

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    get_preprocessed_data(
        args.input_file,
    )
