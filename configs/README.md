# Configuration examples

`video.example.json` and `rtsp.example.json` are consumed by the root
`predict_video.py` command using `--config`. Paths in these JSON files are
relative to the project root. Command-line options override JSON values.

The RTSP example specifies an environment-variable name, not a camera URL.
Set `P2PNET_RTSP_URL` using your local environment settings; the parser rejects
credential-bearing `source` values in JSON config files.

Image inference (`inference/run_frames.py`) and the upstream training script
currently use command-line options rather than these JSON files.
