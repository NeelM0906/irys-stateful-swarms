# blackboard-mcp (Claude Code plugin)

Self-contained Claude Code plugin wrapping [`@irys/blackboard-mcp`](../README.md) — persistent, provenance-tracked blackboard reasoning for AI agents. No API keys, no external services.

This bundle ships the server as a single dependency-free file (`server/index.mjs`, built with esbuild — `@modelcontextprotocol/sdk` and `zod` inlined), so installing it does **not** require cloning the repo, running `npm install`, or building TypeScript, and there's no `node_modules/` tree in the archive. It only requires `node` (>=18) on PATH.

Earlier builds shipped a raw `node_modules/` folder instead. That's what kept failing "Upload as local plugin" with a generic "Zip file contains path with invalid characters" error — the nested `@modelcontextprotocol` scoped-package directory and thousands of third-party files gave the upload validator plenty of surface area to choke on, and pruning individual files didn't fix it. Bundling to one file removes the whole dependency tree, not just the parts we guessed were the problem.

## Try it locally

```bash
claude --plugin-dir ./blackboard-mcp-plugin.zip
```

or point at the unzipped folder directly:

```bash
claude --plugin-dir ./blackboard-mcp
```

Confirm it loaded: ask Claude to "list all blackboards" — that calls `bb_list`.

## Install for real use

Unzip into wherever you keep local plugins, or drop the folder into a private marketplace repo (`marketplace.json` referencing this directory as a `local` source) so the rest of the team can `/plugin install blackboard-mcp` instead of hand-copying files. See the [Claude Code plugin marketplace docs](https://code.claude.com/docs/en/plugin-marketplaces) for the marketplace step — not set up yet, this bundle is just the installable unit.

## Rebuilding this bundle

From `packages/blackboard-mcp/`:

```bash
claude-plugin/build.sh
```

Produces `claude-plugin/dist/blackboard-mcp-plugin-VERSION.zip` (VERSION taken from `package.json`). Re-run after any change to `src/` or a dependency bump — the zip is a build artifact, not checked into git.

## What's inside

```
blackboard-mcp/
├── .claude-plugin/plugin.json   # plugin metadata
├── .mcp.json                    # registers the "blackboard" MCP server
├── server/index.mjs             # esbuild bundle: bb_* tools + all dependencies inlined
├── LICENSE
└── README.md                    # this file
```
