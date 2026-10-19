#!/bin/bash
# 停止所有训练相关进程：先杀链条管理器，再杀训练
# ⚠️ 用 comm 字段精确匹配 python，避免匹配到执行本脚本的 shell 自身
cd /home/su127/FYP/domain-bounded-cslr || exit 1

echo "=== 1. 停链条管理器（否则它会自动重启训练）==="
for p in $(ps -eo pid,comm --no-headers | awk '$2=="python" {print $1}'); do
    if tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q "p140_chain"; then
        echo "  kill chain pid=$p"
        kill "$p" 2>/dev/null
    fi
done
sleep 3

echo "=== 2. 停训练进程 ==="
for p in $(ps -eo pid,comm --no-headers | awk '$2=="python" {print $1}'); do
    if tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q "p78_train"; then
        echo "  kill train pid=$p"
        kill "$p" 2>/dev/null
    fi
done
sleep 5

echo "=== 3. 强制清理残留 ==="
for p in $(ps -eo pid,comm --no-headers | awk '$2=="python" {print $1}'); do
    if tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -qE "p140_chain|p78_train"; then
        echo "  kill -9 pid=$p"
        kill -9 "$p" 2>/dev/null
    fi
done
sleep 3

echo "=== 剩余进程 ==="
ps -eo pid,comm,args --no-headers | grep -E "p140_chain|p78_train" | grep -v grep | wc -l

echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

echo "=== 权重完好性（应 >1MB，MEMORY 22 残file 防护）==="
ls -la artifacts/checkpoints/p139-lr25e5-*.pt 2>/dev/null
