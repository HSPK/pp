"""Coding-agent operating modes.

Ported from `packages/coding-agent/src/modes/` in the TypeScript "pi"
monorepo. Only the RPC transport surface is represented here
(`modes.rpc`); the legacy stdio JSON-line embedding mode
(`modes/rpc/rpc-mode.ts` and `modes/rpc/rpc-client.ts` in the source tree,
despite the `rpc` directory name) was NOT ported. See `modes.rpc`'s
docstring for why.
"""
