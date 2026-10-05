#!/usr/bin/env bash
REPO=/home/su127/FYP/domain-bounded-cslr
echo "== data/raw/CE-CSL tree (depth 2) =="
find "$REPO/data/raw/CE-CSL" -maxdepth 2 -type d 2>/dev/null | head -40
echo
echo "== sizes =="
du -sh "$REPO/data/raw/CE-CSL" 2>/dev/null
du -sh "$REPO/artifacts/part3_features" 2>/dev/null
du -sh "$REPO/data/processed" 2>/dev/null
echo
echo "== largest files under data/raw/CE-CSL =="
find "$REPO/data/raw/CE-CSL" -type f 2>/dev/null | xargs du -h 2>/dev/null | sort -rh | head -5
echo
echo "== count mp4 under CE-CSL =="
find "$REPO/data/raw/CE-CSL" -iname '*.mp4' 2>/dev/null | wc -l