# Licensing and publishing checklist

## Licensing

- `original_repo/LICENSE` is the license distributed with the upstream
  Tencent P2PNet implementation. It limits use of that source and binary
  distributions to **academic research** and requires retaining its notice and
  terms. It is not a general-purpose permissive or commercial-use license.
- The local `SHTechA.pth` checkpoint matches the file in the official upstream
  repository. Its distribution/use remains subject to the upstream terms; this
  project ignores checkpoint files by default.
- The JHU-CROWD++ `License` allows academic and non-commercial use only and
  prohibits commercial use of the images and derived data. Raw JHU data is
  excluded from Git.
- No project-wide license has been selected for the additions in this
  repository. Do not add a root `LICENSE` file until the owner has established
  the rights and desired terms for every included component.
- Obtain appropriate rights and consent for user-provided park videos,
  camera footage, and annotations before publishing or processing them.

This note is an inventory of the terms present in the project, not legal
advice. Confirm with the relevant rights holders before public redistribution
or any non-academic/commercial use.

## Before the first push

1. Confirm the repository is intended for academic/research use under the
   upstream P2PNet terms.
2. Review `git status --short` and `git diff --cached --stat`; `.gitignore`
   excludes virtual environments, checkpoints, raw datasets, camera videos,
   and generated outputs by default.
3. Check staged files for URLs containing credentials, `.env` files, local
   paths, private footage, and generated CSV/JSON/video results.
4. Do not force-add `*.pth` unless redistribution is approved and the required
   notices will be included. If approved, use Git LFS and verify account quota.
5. Add a root license only after selecting terms for the project's own code
   and reconciling them with the upstream restrictions.
6. This project directory currently has no `.git` metadata. Create/commit/push
   only after reviewing the staged file list. Do not push data or weights
   merely because they exist locally.
