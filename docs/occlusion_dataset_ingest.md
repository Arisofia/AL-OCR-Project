# Occlusion Dataset Ingest

Use these local folders for real-image ingestion and annotation prep.

## Upload locations

- Raw uploads (training candidates): `data/raw/occlusion_cards/inbox`
- Holdout set (do not train on these): `data/raw/occlusion_cards/holdout`
- Validated/cleaned samples: `data/validated/occlusion_cards`
- Label Studio working dir (optional): `data/label-studio/occlusion_cards`

## Recommended file naming

`occl_<source>_<yyyy-mm-dd>_<id>.jpg`

Example:

`occl_mobile_2026-02-17_0001.jpg`

## Safety and compliance

- Do not upload full PAN datasets to Git.
- Keep real card images local or in secured storage only.
- Prefer masked/partially redacted samples for model iteration.

## Metadata template

Use `data/occlusion_manifest_template.csv` to track labels and quality.
