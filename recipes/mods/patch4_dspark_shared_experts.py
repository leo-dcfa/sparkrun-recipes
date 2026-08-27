# Patch 4 from tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark
# (patches/0004-dspark-shared-expert-gate-up-proj.patch), applied programmatically.
# Adds the draft shared-expert gate_up_proj w1/w3 stacked-param mappings that the
# DSpark draft loader lost; without them 12 tensors load uninitialised and draft
# acceptance collapses from ~60% to ~26% on the 0731 checkpoint.
import sys

F = "/opt/env/lib/python3.12/site-packages/vllm/v1/spec_decode/dspark.py"
ANCHOR = '    ("attn.fused_wqa_wkv", ".attn.wkv", 1),'
ADD = """    # Patch 4 (tonyd2wild 0731 repo): map the DSpark draft shared expert's
    # gate_up_proj (MergedColumnParallelLinear) from the checkpoint's w1/w3.
    # The loader only renamed .shared_experts.w2 -> down_proj, so w1/w3 were
    # silently dropped, leaving layers {43,44,45}.ffn.shared_experts.gate_up_proj
    # uninitialised in every draft stage (acceptance ~60% -> ~26%).
    ("shared_experts.gate_up_proj", ".shared_experts.w1", 0),
    ("shared_experts.gate_up_proj", ".shared_experts.w3", 1),"""

src = open(F).read()
if "shared_experts.gate_up_proj" in src:
    print("already patched")
    sys.exit(0)
assert ANCHOR in src, "anchor line not found in dspark.py"
open(F, "w").write(src.replace(ANCHOR, ANCHOR + "\n" + ADD, 1))
print("patched")
