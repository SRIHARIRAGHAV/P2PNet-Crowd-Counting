# P2PNet checkpoint

Inference requires the official ShanghaiTech Part A checkpoint:

- Filename: `SHTechA.pth`
- Expected project path: `original_repo/weights/SHTechA.pth`
- Expected size: 86,372,926 bytes
- SHA-256 for the checkpoint used during local verification:
  `506047732B128FF09EFEF18E94BFACBE35FCFEF300E5E9EEEECE259B0488C63F`

The local checkpoint was verified byte-for-byte against the Git blob in the
official Tencent P2PNet repository. The upstream file is available at:

<https://raw.githubusercontent.com/TencentYoutuResearch/CrowdCounting-P2PNet/main/weights/SHTechA.pth>

The upstream P2PNet license permits source/binary redistribution **only for
academic research**, and requires retaining its notice and license terms. This
project excludes checkpoint files from Git by default. Obtain the checkpoint
from the official source only if your use complies with that license; do not
upload it to this project or a release unless you have verified permission and
include the required notices.

On Windows, download into the expected location from the project root:

```powershell
New-Item -ItemType Directory -Force .\original_repo\weights | Out-Null
curl.exe -L `
  "https://raw.githubusercontent.com/TencentYoutuResearch/CrowdCounting-P2PNet/main/weights/SHTechA.pth" `
  -o .\original_repo\weights\SHTechA.pth
Get-FileHash .\original_repo\weights\SHTechA.pth -Algorithm SHA256
```

On Linux/macOS:

```bash
mkdir -p original_repo/weights
curl -L \
  https://raw.githubusercontent.com/TencentYoutuResearch/CrowdCounting-P2PNet/main/weights/SHTechA.pth \
  -o original_repo/weights/SHTechA.pth
sha256sum original_repo/weights/SHTechA.pth
```

Compare the resulting SHA-256 with the value above. Then run a one-frame
inference smoke test from the project root:

```powershell
.\.venv-cpu\Scripts\python.exe .\predict_video.py `
  --input <path-to-a-video> `
  --checkpoint .\original_repo\weights\SHTechA.pth `
  --device cpu `
  --max-frames 1 `
  --output .\outputs\checkpoint_smoke\smoke.mp4
```

The loader uses `map_location` and strict state-dictionary matching. It reports
an error for missing, corrupt, or incompatible checkpoints; it does not fall
back to randomly initialized weights. Select another compatible checkpoint
with `--checkpoint` in `predict_video.py`, or `--weight` in
`inference/run_frames.py`.

The official source license is preserved at `original_repo/LICENSE`. No
project-wide license is asserted by adding a root `LICENSE` file.

## Optional Git LFS

Use Git LFS only after confirming that your redistribution is permitted by the
upstream license and your GitHub LFS quota is sufficient. The checkpoint is
currently ignored by `.gitignore`; an authorized LFS addition would require:

```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
git add -f original_repo/weights/SHTechA.pth
```

Do not use these commands unless redistribution has been approved. A safer
default is to keep the checkpoint out of the repository and have users obtain
it from the official upstream source.
