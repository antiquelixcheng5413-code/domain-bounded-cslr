#!/usr/bin/env bash
for d in /mnt/d/part3_models /mnt/d/models /home/su127/FYP/models /home/su127/FYP/domain-bounded-cslr/artifacts/models; do
  echo "== $d =="
  ls "$d" 2>/dev/null || echo "(missing)"
done
echo "== find qwen llm dirs =="
find /mnt/d /home/su127/FYP -maxdepth 4 -iname '*qwen*' -type d 2>/dev/null | head -20
echo "== find gloss->llm receipts for prior model refs =="
grep -rl "Qwen\|qwen" /home/su127/FYP/domain-bounded-cslr/artifacts/metrics/*route_a*.json 2>/dev/null | head