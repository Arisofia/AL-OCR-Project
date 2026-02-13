#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "[cicd] Linting workflow YAML"
yamllint -c .yamllint .github/workflows .github/dependabot.yml

echo "[cicd] Parsing workflow YAML"
python3 - <<'PY'
from pathlib import Path
import yaml

for path in list(Path('.github/workflows').glob('*')) + [Path('.github/dependabot.yml')]:
    if not path.is_file():
        continue
    yaml.safe_load(path.read_text())
print("workflow-yaml-parse-ok")
PY

echo "[cicd] Running static workflow sanity checks"
python3 - <<'PY'
from pathlib import Path
import re

workflows = list(Path('.github/workflows').glob('*'))
errors: list[str] = []

for wf in workflows:
    if not wf.is_file():
        continue
    text = wf.read_text()

    if 'ocr-service/**' in text or './ocr-service' in text:
        errors.append(f"{wf}: references non-existent path ocr-service; use ocr_service")

    if re.search(r'uses:\s*[^@\s]+@(main|master|HEAD)\b', text):
        errors.append(f"{wf}: uses floating action ref (main/master/HEAD); pin to a version")

if errors:
    for err in errors:
        print(err)
    raise SystemExit(1)

print('workflow-static-sanity-ok')
PY

echo "[cicd] CI/CD workflow verification complete"
