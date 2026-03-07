#!/bin/bash
set -e

# Fix: Qwen3NextSparseMoeBlock creates the MoE gate (router) with
# quant_config=quant_config, which makes it expect quantized parameters
# (qweight/qzeros/scales). But the checkpoint stores gate weights as
# plain BF16 gate.weight. This causes gate weights to be skipped during
# loading, breaking MoE routing and producing garbage output.
#
# Fix: Change gate creation to use quant_config=None.

SITE_PACKAGES="/usr/local/lib/python3.12/dist-packages"
TARGET="$SITE_PACKAGES/vllm/model_executor/models/qwen3_next.py"

if [ ! -f "$TARGET" ]; then
    echo "[fix-qwen3.5-moe-gate] qwen3_next.py not found, skipping"
    exit 0
fi

# Check if already fixed (gate uses quant_config=None)
if python3 -c "
import re
with open('$TARGET') as f:
    code = f.read()
# Find the gate = ReplicatedLinear block and check if it already uses quant_config=None
m = re.search(r'self\.gate\s*=\s*ReplicatedLinear\([^)]*quant_config=None[^)]*\)', code, re.DOTALL)
if m:
    exit(0)
else:
    exit(1)
" 2>/dev/null; then
    echo "[fix-qwen3.5-moe-gate] Already fixed, skipping"
    exit 0
fi

echo "[fix-qwen3.5-moe-gate] Patching gate quant_config in Qwen3NextSparseMoeBlock..."

python3 << 'PYEOF'
import re

path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py"
with open(path) as f:
    code = f.read()

# Find: self.gate = ReplicatedLinear(..., quant_config=quant_config, ...)
# Replace quant_config=quant_config with quant_config=None for the gate only
#
# We need to be careful to only change the self.gate assignment, not shared_expert_gate
# or any other ReplicatedLinear.

# Match the self.gate = ReplicatedLinear(...) block
pattern = r'(self\.gate\s*=\s*ReplicatedLinear\([^)]*?)quant_config=quant_config([^)]*?\))'
replacement = r'\1quant_config=None\2'

new_code, count = re.subn(pattern, replacement, code, count=1, flags=re.DOTALL)

if count == 0:
    print("WARNING: Could not find self.gate = ReplicatedLinear(...quant_config=quant_config...) pattern")
    print("Trying alternative approach...")

    # Try line-by-line approach
    lines = code.split('\n')
    in_gate_block = False
    modified = False
    for i, line in enumerate(lines):
        if 'self.gate = ReplicatedLinear(' in line:
            in_gate_block = True
        if in_gate_block and 'quant_config=quant_config' in line:
            lines[i] = line.replace('quant_config=quant_config', 'quant_config=None')
            in_gate_block = False
            modified = True
            print(f"Fixed line {i+1}: {lines[i].strip()}")
            break
        if in_gate_block and ')' in line and 'quant_config' not in line:
            # End of ReplicatedLinear call without finding quant_config
            in_gate_block = False

    if modified:
        new_code = '\n'.join(lines)
    else:
        print("ERROR: Could not apply fix")
        import sys
        sys.exit(1)
else:
    print(f"Fixed {count} occurrence(s) of quant_config in self.gate")

with open(path, 'w') as f:
    f.write(new_code)

print("[fix-qwen3.5-moe-gate] Patch applied successfully!")
PYEOF
