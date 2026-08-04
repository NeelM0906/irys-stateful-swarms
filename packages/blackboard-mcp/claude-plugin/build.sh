#!/usr/bin/env bash
# Builds packages/blackboard-mcp into a self-contained Claude Code plugin.
#
# The output is written IN PLACE, directly into this directory
# (claude-plugin/server/index.mjs), and is meant to be committed to git.
# That's what makes marketplace.json's "./packages/blackboard-mcp/claude-plugin"
# source installable via /plugin install — a marketplace entry has to point
# at something runnable, and a gitignored build artifact isn't. The zip under
# dist/ is a separate, still-gitignored convenience export for the manual
# "Upload as local plugin" / --plugin-dir <zip> path.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../packages/blackboard-mcp/claude-plugin
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"                       # .../packages/blackboard-mcp
REPO_ROOT="$(cd "$PKG_DIR/../.." && pwd)"

OUT_DIR="$SCRIPT_DIR/dist"

echo "==> Installing build tooling"
cd "$PKG_DIR"
npm install

VERSION="$(node -p "require('./package.json').version")"
echo "==> Building version $VERSION"

# LICENSE is refreshed from repo root each build; everything else
# (.claude-plugin/, .mcp.json, README.md) is already source-of-truth here.
cp "$REPO_ROOT/LICENSE" "$SCRIPT_DIR/LICENSE"

# Keep plugin.json version in sync with package.json (single source of truth).
node -e "
  const fs = require('fs');
  const p = '$SCRIPT_DIR/.claude-plugin/plugin.json';
  const j = JSON.parse(fs.readFileSync(p, 'utf-8'));
  j.version = '$VERSION';
  fs.writeFileSync(p, JSON.stringify(j, null, 2) + '\n');
"

# Bundle the server into a single dependency-free file instead of shipping a
# node_modules/ tree. A raw node_modules folder (nested @scope directories,
# thousands of entries, deep paths from third-party packages) is what kept
# tripping the plugin-zip upload validator with "invalid characters" even
# after pruning individual files — see claude-plugin/README.md. esbuild
# inlines @modelcontextprotocol/sdk + zod + transitives into one .mjs file,
# so there's no dependency tree left in the archive at all.
echo "==> Bundling server (esbuild, single file, no node_modules)"
mkdir -p "$SCRIPT_DIR/server"
npx --yes esbuild@0.28.1 "$PKG_DIR/src/index.ts" \
  --bundle --platform=node --format=esm --target=node18 \
  --outfile="$SCRIPT_DIR/server/index.mjs"

echo "==> Zipping (convenience export for manual upload, not committed)"
mkdir -p "$OUT_DIR"
ZIP_NAME="blackboard-mcp-plugin-$VERSION.zip"
rm -f "$OUT_DIR/$ZIP_NAME"
# Zip claude-plugin/'s contents at the archive root (no wrapping folder, and
# excluding dist/ itself and this script), so .claude-plugin/plugin.json
# sits at the zip root as --plugin-dir expects.
(cd "$SCRIPT_DIR" && zip -qr "$OUT_DIR/$ZIP_NAME" . -x "dist/*" -x "build.sh")

echo ""
echo "Done."
echo "  Committed plugin dir: $SCRIPT_DIR  (git add this)"
echo "  Zip archive:          $OUT_DIR/$ZIP_NAME  (gitignored, for manual upload)"
echo ""
echo "Test with: claude --plugin-dir $SCRIPT_DIR"
