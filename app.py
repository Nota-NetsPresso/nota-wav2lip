import os
import subprocess
from pathlib import Path

import gradio as gr

from config import hparams as hp
from config import hparams_gradio as hp_gradio
from nota_wav2lip import Wav2LipModelComparisonGradio

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = hp_gradio.device
print(f'Using {device} for inference.')
video_label_dict = hp_gradio.sample.video
audio_label_dict = hp_gradio.sample.audio

LRS_ORIGINAL_URL = os.getenv('LRS_ORIGINAL_URL', None)
LRS_COMPRESSED_URL = os.getenv('LRS_COMPRESSED_URL', None)
LRS_INFERENCE_SAMPLE = os.getenv('LRS_INFERENCE_SAMPLE', None)

if not Path(hp.inference.model.wav2lip.checkpoint).exists() and LRS_ORIGINAL_URL is not None:
    subprocess.call(f"wget --no-check-certificate -O {hp.inference.model.wav2lip.checkpoint} {LRS_ORIGINAL_URL}", shell=True)
if not Path(hp.inference.model.nota_wav2lip.checkpoint).exists() and LRS_COMPRESSED_URL is not None:
    subprocess.call(f"wget --no-check-certificate -O {hp.inference.model.nota_wav2lip.checkpoint} {LRS_COMPRESSED_URL}", shell=True)

path_inference_sample = "sample.tar.gz"
if not Path(path_inference_sample).exists() and LRS_INFERENCE_SAMPLE is not None:
    subprocess.call(f"wget --no-check-certificate -O {path_inference_sample} {LRS_INFERENCE_SAMPLE}", shell=True)
subprocess.call(f"tar -zxvf {path_inference_sample}", shell=True)


if __name__ == "__main__":

    servicer = Wav2LipModelComparisonGradio(
        device=device,
        video_label_dict=video_label_dict,
        audio_label_list=audio_label_dict,
        default_video='v1',
        default_audio='a1'
    )

    for video_name in sorted(video_label_dict):
        video_stem = Path(video_label_dict[video_name])
        servicer.update_video(video_stem, video_stem.with_suffix('.json'),
                              name=video_name)

    for audio_name in sorted(audio_label_dict):
        audio_path = Path(audio_label_dict[audio_name])
        servicer.update_audio(audio_path, name=audio_name)

    with gr.Blocks(theme='nota-ai/theme', css=Path('docs/main.css').read_text()) as demo:
        gr.Markdown(Path('docs/header.md').read_text())
        gr.Markdown(Path('docs/description.md').read_text())
        with gr.Row():
            with gr.Column(variant='panel'):

                gr.Markdown('## Select input video and audio', sanitize_html=False)
                # Define samples
                sample_video = gr.Video(interactive=False, label="Input Video")
                sample_audio = gr.Audio(interactive=False, label="Input Audio")

                # Define radio inputs
                video_selection = gr.components.Radio(video_label_dict,
                                                      type='value', label="Select an input video:")
                audio_selection = gr.components.Radio(audio_label_dict,
                                                      type='value', label="Select an input audio:")
                # Define button inputs
                with gr.Row(equal_height=True):
                    generate_original_button = gr.Button(value="Generate with Original Model", variant="primary")
                    generate_compressed_button = gr.Button(value="Generate with Compressed Model", variant="primary")
            with gr.Column(variant='panel'):
                # Define original model output components
                gr.Markdown('## Original Wav2Lip')
                original_model_output = gr.Video(label="Original Model", interactive=False)
                with gr.Column():
                    with gr.Row(equal_height=True):
                        original_model_inference_time = gr.Textbox(value="", label="Total inference time (sec)")
                        original_model_fps = gr.Textbox(value="", label="FPS")
                    original_model_params = gr.Textbox(value=servicer.params['wav2lip'], label="# Parameters")
            with gr.Column(variant='panel'):
                # Define compressed model output components
                gr.Markdown('## Compressed Wav2Lip (Ours)')
                compressed_model_output = gr.Video(label="Compressed Model", interactive=False)
                with gr.Column():
                    with gr.Row(equal_height=True):
                        compressed_model_inference_time = gr.Textbox(value="", label="Total inference time (sec)")
                        compressed_model_fps = gr.Textbox(value="", label="FPS")
                    compressed_model_params = gr.Textbox(value=servicer.params['nota_wav2lip'], label="# Parameters")

        # Switch video and audio samples when selecting the raido button
        video_selection.change(fn=servicer.switch_video_samples, inputs=video_selection, outputs=sample_video)
        audio_selection.change(fn=servicer.switch_audio_samples, inputs=audio_selection, outputs=sample_audio)

        # Click the generate button for original model
        generate_original_button.click(servicer.generate_original_model,
                                       inputs=[video_selection, audio_selection],
                                       outputs=[original_model_output, original_model_inference_time, original_model_fps])
        # Click the generate button for compressed model
        generate_compressed_button.click(servicer.generate_compressed_model,
                                         inputs=[video_selection, audio_selection],
                                         outputs=[compressed_model_output, compressed_model_inference_time, compressed_model_fps])

        gr.Markdown(Path('docs/footer.md').read_text())

    demo.queue().launch()
