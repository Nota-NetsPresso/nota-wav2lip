import threading
from pathlib import Path

from nota_wav2lip.demo import Wav2LipModelComparisonDemo


class Wav2LipModelComparisonGradio(Wav2LipModelComparisonDemo):
    def __init__(
        self,
        device='cpu',
        result_dir='./temp',
        video_label_dict=None,
        audio_label_list=None,
        default_video='v1',
        default_audio='a1'
    ) -> None:
        if audio_label_list is None:
            audio_label_list = {}
        if video_label_dict is None:
            video_label_dict = {}
        super().__init__(device, result_dir)
        self._video_label_dict = {k: Path(v).with_suffix('.mp4') for k, v in video_label_dict.items()}
        self._audio_label_dict = audio_label_list
        self._default_video = default_video
        self._default_audio = default_audio

        self._lock = threading.Lock()  # lock for asserting that concurrency_count == 1

    def _is_valid_input(self, video_selection, audio_selection):
        assert video_selection in self._video_label_dict, \
            f"Your input ({video_selection}) is not in {self._video_label_dict}!!!"
        assert audio_selection in self._audio_label_dict, \
            f"Your input ({audio_selection}) is not in {self._audio_label_dict}!!!"

    def generate_original_model(self, video_selection, audio_selection):
        try:
            self._is_valid_input(video_selection, audio_selection)

            with self._lock:
                output_video_path, inference_time, inference_fps = \
                    self.save_as_video(audio_name=audio_selection,
                                       video_name=video_selection,
                                       model_type='wav2lip')

                return str(output_video_path), format(inference_time, ".2f"), format(inference_fps, ".1f")
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(e)
            pass

    def generate_compressed_model(self, video_selection, audio_selection):
        try:
            self._is_valid_input(video_selection, audio_selection)

            with self._lock:
                output_video_path, inference_time, inference_fps = \
                    self.save_as_video(audio_name=audio_selection,
                                       video_name=video_selection,
                                       model_type='nota_wav2lip')

                return str(output_video_path), format(inference_time, ".2f"), format(inference_fps, ".1f")
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(e)
            pass

    def switch_video_samples(self, video_selection):
        try:
            if video_selection not in self._video_label_dict:
                return self._video_label_dict[self._default_video]
            return self._video_label_dict[video_selection]

        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(e)
            pass

    def switch_audio_samples(self, audio_selection):
        try:
            if audio_selection not in self._audio_label_dict:
                return self._audio_label_dict[self._default_audio]
            return self._audio_label_dict[audio_selection]

        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(e)
            pass
