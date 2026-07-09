#!/usr/bin/env python3
"""Default the hy_v3 parser token_suffix to ':opensource'.

Why: kodelow/Hy3-NVFP4-W4A16's tokenizer_config.json does NOT set
init_kwargs["token_suffix"], so vLLM's dynamic hy_v3 reasoning/tool parsers fall
back to bare <think>/<tool_call> tokens and crash at startup ("could not locate
think start/end tokens"). This rewrites the empty-string fallback to ":opensource"
in both parser files, in place, inside the image.

This is the reproducible form of the one-line patch that originally produced
eugr/spark-vllm:hy3-opensource (whose Dockerfile was lost to an ephemeral
scratchpad). It targets the most likely code shape:  ...get("token_suffix", "")...
and FAILS LOUDLY if it can't find that pattern — if the build fails here, inspect
the two files inside eugr/spark-vllm:latest and update SUFFIX_RE / the default
below (see README.md, "hy3-295b-nvfp4 — caveats", runbook step B).
"""
import pathlib
import re
import sys

import vllm

SENTINEL = ":opensource"
# Matches  "token_suffix", ""   or   'token_suffix', ''   (an empty-string default).
SUFFIX_RE = re.compile(r"""(['"]token_suffix['"]\s*,\s*)(['"])\2""")

root = pathlib.Path(vllm.__file__).parent
targets = sorted(
    set(root.glob("**/hy_v3_reasoning_parser.py")) | set(root.glob("**/hy_v3_tool_parser.py"))
)
if not targets:
    sys.exit(f"ERROR: no hy_v3 parser files found under {root}")

patched = 0
for f in targets:
    text = f.read_text()
    new = SUFFIX_RE.sub(rf'\1"{SENTINEL}"', text)
    if new != text:
        f.write_text(new)
        patched += 1
        print(f"patched: {f}")
    else:
        print(f"no empty-string token_suffix default matched in: {f}")

if patched == 0:
    sys.exit(
        "ERROR: nothing patched — the parser source doesn't match the assumed "
        "pattern. Inspect the two hy_v3 parser files and update this script "
        "(README runbook step B)."
    )

# Verify the sentinel is actually present in every target now.
for f in targets:
    if SENTINEL not in f.read_text():
        sys.exit(f"ERROR: verification failed — {SENTINEL} missing from {f}")

print(f"OK: token_suffix defaulted to '{SENTINEL}' in {patched} file(s)")
