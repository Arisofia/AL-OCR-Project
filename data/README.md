# Data workspace layout

This repository tracks metadata in `data/` and keeps bulky/generated datasets out of Git.

Ignored working directories (created locally as needed):
- `data/raw/`
- `data/validated/`
- `data/label-studio/`

Recommended local folder scaffold for occlusion workflow:

```bash
mkdir -p data/raw/occlusion_cards/{inbox,holdout} \
         data/validated/occlusion_cards \
         data/label-studio/occlusion_cards
```

Notes:
- `data/raw.dvc` controls raw dataset artifacts through DVC.
- Keep secrets and personal/sensitive files out of `data/`.
