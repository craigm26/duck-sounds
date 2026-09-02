#!/usr/bin/env bash
# Upload the Microduck Stairs Challenge package to the Hugging Face Hub as a DATASET.
#
# THE OPERATOR LOGS IN FIRST. This script does not authenticate and will not
# prompt for a token. Run `hf auth login` (or export HF_TOKEN) in your own shell
# before running this, then set HF_USER to the account or org that should own
# the dataset.
#
#   HF_USER=your-hf-username ./hf_upload.sh
#
set -euo pipefail

HF_USER="${HF_USER:?set HF_USER to the Hugging Face account or org that will own the dataset}"
REPO="${HF_USER}/microduck-stairs-challenge"
SRC="/home/craigm26/projects/duck-sounds/challenge"

# 1. Who am I? Fails loudly if the operator has not logged in.
hf auth whoami

# 2. Create the dataset repo. Idempotent: --exist-ok makes a re-run a no-op.
hf repo create "${REPO}" --type dataset --exist-ok

# 3. Push the whole package.
hf upload "${REPO}" "${SRC}" . \
  --repo-type dataset \
  --commit-message "Microduck Stairs Challenge v1"

echo
echo "Done: https://huggingface.co/datasets/${REPO}"
