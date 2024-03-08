# Original Wav2Lip
python inference.py\
  -a "sample_video_lrs3/sxnlvwprf_c-00007.wav"\
  -v "sample_video_lrs3/Li4-1yyrsTI-00010"\
  -m "wav2lip"\
  -o "result_original"\
  --device cpu

# Nota's Wav2Lip (28× Compressed)
python inference.py\
  -a "sample_video_lrs3/sxnlvwprf_c-00007.wav"\
  -v "sample_video_lrs3/Li4-1yyrsTI-00010"\
  -m "nota_wav2lip"\
  -o "result_nota"\
  --device cpu 