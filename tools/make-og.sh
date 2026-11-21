#!/usr/bin/env bash
# Render the social sharing card site/og.html to docs/og.png (1200x630) with headless Chromium.
#   tools/make-og.sh        (uses `chromium` on PATH, or nix-shell -p chromium)
set -euo pipefail
cd "$(dirname "$0")/.."
args=(--headless=new --no-sandbox --hide-scrollbars --window-size=1200,630 --virtual-time-budget=8000
      --screenshot="$PWD/docs/og.png" "file://$PWD/site/og.html")
if command -v chromium >/dev/null; then chromium "${args[@]}" 2>/dev/null
else nix-shell -p chromium --run "chromium $(printf '%q ' "${args[@]}") 2>/dev/null"; fi
echo "wrote docs/og.png"
