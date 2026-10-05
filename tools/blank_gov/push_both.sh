#!/usr/bin/env bash
# 推送当前分支到**两个**远端，并验证两边都收到。
#
# 🔴 为什么需要这个脚本（2026-10-05）：
#   本仓库配了两个远端，我曾两次只推 origin 就报告「已推送」，
#   队友反馈「没收到」才发现。推一个远端等于没推。
#
# 用法：
#   bash tools/blank_gov/push_both.sh                     # 推当前分支
#   bash tools/blank_gov/push_both.sh experiment/xxx      # 推指定分支
#   DRY_RUN=1 bash tools/blank_gov/push_both.sh           # 只看要推什么，不真推

set -euo pipefail

BRANCH="${1:-$(git branch --show-current)}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "仓库    : $REPO_ROOT"
echo "分支    : $BRANCH"
echo "提交    : $(git log --oneline -1)"
echo

# ---- 提交前安全检查 ----
echo "=== 提交前检查 ==="
if [ -z "$(git status --porcelain)" ]; then
  echo "  工作区干净 ✓"
else
  echo "  ⚠️ 有未提交改动："
  git status --short | head -10
  echo "  （不阻塞推送，但请确认这些改动是否该提交）"
fi

# 大文件检查（特征/checkpoint 目录）
BIG=$(git status --porcelain | awk '{print $2}' | while read -r f; do
  [ -f "$f" ] && du -k "$f" 2>/dev/null
done | sort -rn | head -3)
if [ -n "$BIG" ]; then
  echo "  待提交文件体积 TOP3："
  echo "$BIG" | sed 's/^/    /'
fi

IGNORED_OK=0
for d in artifacts/part3_features artifacts/part3_features_tasksapi \
         artifacts/checkpoints; do
  # ⚠️ 必须用带通配的路径检查：`git check-ignore` 对**已存在**的目录才能命中，
  #    而 part3_features_tasksapi 可能在磁盘上还不存在（首次运行时），
  #    直接传目录名会误报「未被忽略」。
  #    用 `dir/**` 匹配规则本身，与目录是否存在无关。
  if git check-ignore -q "$d/**" 2>/dev/null || \
     grep -qE "^${d}/\*\*" .gitignore 2>/dev/null; then
    IGNORED_OK=$((IGNORED_OK + 1))
  else
    echo "  ⚠️ $d 未被 gitignore —— 可能误提交大文件！"
    echo "     修复：.gitignore 加一行  ${d}/**"
  fi
done
[ "$IGNORED_OK" -eq 3 ] && echo "  大目录忽略规则正常 ✓（3/3）" \
                      || echo "  ⚠️ $IGNORED_OK/3 个大目录有忽略规则"
echo

# ---- 远端状态 ----
echo "=== 远端状态 ==="
for r in origin shared; do
  if git remote get-url "$r" >/dev/null 2>&1; then
    URL=$(git remote get-url "$r")
    AHEAD=$(git rev-list --count "$r/$BRANCH..HEAD" 2>/dev/null || echo "?")
    echo "  $r  ($URL)"
    echo "      本地领先: $AHEAD 个提交"
  else
    echo "  $r  ⚠️ 未配置"
  fi
done
echo

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN，未实际推送"
  exit 0
fi

# ---- 推送 ----
FAILED=0
for r in origin shared; do
  if ! git remote get-url "$r" >/dev/null 2>&1; then
    echo "跳过 $r（未配置）"
    continue
  fi
  echo "=== 推送到 $r ==="
  if git push "$r" "$BRANCH"; then
    echo "  $r ✓"
  else
    echo "  $r ✗ 推送失败"
    FAILED=1
  fi
  echo
done

# ---- 验证两个远端都收到了 ----
echo "=== 验证 ==="
SHA=$(git rev-parse HEAD)
OK=1
for r in origin shared; do
  if git remote get-url "$r" >/dev/null 2>&1; then
    if git ls-remote "$r" "refs/heads/$BRANCH" 2>/dev/null | grep -q "$SHA"; then
      echo "  $r  ✓ 已包含 $SHA"
    else
      echo "  $r  ✗ 未找到 $SHA —— 推送未生效！"
      OK=0
    fi
  fi
done

if [ "$OK" = "1" ] && [ "$FAILED" = "0" ]; then
  echo
  echo "✅ 两个远端均已同步"
  echo "   PR 链接（shared）:"
  echo "   https://github.com/antiquelixcheng5413-code/domain-bounded-cslr/pull/new/$BRANCH"
else
  echo
  echo "❌ 有远端未同步，请检查上面的错误"
  exit 1
fi
