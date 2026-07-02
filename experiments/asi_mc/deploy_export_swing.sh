#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Step 1 of SwingDim pipeline: Export swing indicators via curator-
# identical SwingStructure class.
#
# What this does:
#   1. Create cx53 server + attach neat-data volume (has S5 parquets)
#   2. Mount volume, copy S5 data to local disk
#   3. Copy existing asi_mc_indicators parquets (LOCAL → server)
#   4. Copy lib/indicators.py, lib/swing_indicators.py, export script
#   5. Run export_swing_indicators.py
#      - Uses SwingStructure (same class as live curator)
#      - Adds sb_a, erp_a, hh_asi, hl_asi, erp_p, hh_price, hl_price,
#        d_erp_p, d_erp_a to each parquet
#      - Prints IC pre-screening report for each pair
#   6. Copy augmented parquets BACK to local data/asi_mc_indicators/
#   7. Detach neat-data volume, delete server
#
# After this completes, run deploy_swingdim.sh to launch training.
#
# Cost: 1 × cx53 × ~30min × $0.10/hr = ~$0.05
# ═══════════════════════════════════════════════════════════════
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$REPO_DIR/data/asi_mc_indicators"
LOG="$SCRIPT_DIR/export_swing.log"

VOLUME_NAME="neat-data"
VOLUME_MOUNT="/dev/disk/by-id/scsi-0HC_Volume_105213043"
MOUNT_POINT="/mnt/neat-data"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

echo "" | tee -a "$LOG"
log "═══ SwingDim Export: swing indicators via SwingStructure ═══"
log "Adds: sb_a, erp_a, hh_asi, hl_asi, erp_p, hh_price, hl_price, d_erp_p, d_erp_a"
log "Code: lib/indicators.py:SwingStructure (same as live curator)"
echo ""

# ── 1. Create server ──────────────────────────────────────────
log "Creating cx53 export server..."
# hcloud -o json sends "Waiting for..." status lines to stdout before JSON — grep for JSON line
_create_server() {
    local NAME=$1 TYPE=$2 LOC=$3
    hcloud server create --name "$NAME" --type "$TYPE" --image ubuntu-24.04 \
        --location "$LOC" --ssh-key "user@host" -o json 2>&1 | \
        grep -o '{.*}' | tail -1 | \
        python3 -c "import sys,json; d=sys.stdin.read().strip(); print(json.loads(d)['server']['public_net']['ipv4']['ip']) if d else None" 2>/dev/null || true
}

SERVER_IP=$(_create_server "swingdim-export" "cx53" "hel1")
if [ -z "$SERVER_IP" ]; then
    log "hel1 unavailable, trying nbg1..."
    SERVER_IP=$(_create_server "swingdim-export" "cx53" "nbg1")
fi

log "Server: swingdim-export @ $SERVER_IP"
echo "$SERVER_IP" > "$SCRIPT_DIR/export_server.txt"

# ── 2. Attach neat-data volume ─────────────────────────────────
log "Attaching $VOLUME_NAME volume..."
hcloud volume attach "$VOLUME_NAME" --server swingdim-export --automount=false
log "Volume attached"

# ── 3. Wait for SSH ─────────────────────────────────────────────
log "Waiting for SSH..."
for attempt in $(seq 1 40); do
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=4 root@"$SERVER_IP" 'echo ok' 2>/dev/null && break
    sleep 3
done
log "SSH ready"

# ── 4. Setup server ─────────────────────────────────────────────
log "Installing Python deps + mounting volume..."
ssh -o StrictHostKeyChecking=no root@"$SERVER_IP" "
    apt-get update -qq > /dev/null &&
    apt-get install -y -qq python3-pip python3-venv > /dev/null 2>&1 &&
    python3 -m venv /root/venv &&
    source /root/venv/bin/activate &&
    pip install -q neat-python numba pandas pyarrow numpy requests &&
    mkdir -p /root/neat/data/s5 /root/neat/data/indicators /root/neat/lib &&
    mount '$VOLUME_MOUNT' '$MOUNT_POINT' &&
    echo 'Volume mounted: '$(ls '$MOUNT_POINT' | head -3)'...'
"
log "Setup complete"

# ── 5. Copy S5 parquets from volume to local disk ──────────────
log "Copying S5 parquets from volume → server disk..."
ssh -o StrictHostKeyChecking=no root@"$SERVER_IP" "
    cp '$MOUNT_POINT'/*_S5_BA.parquet /root/neat/data/s5/ 2>/dev/null &&
    echo 'S5 parquets copied: '$(ls /root/neat/data/s5/ | wc -l)' files'
"
log "S5 data on server disk"

# ── 6. Transfer local indicator parquets → server ─────────────
log "Uploading existing asi_mc_indicators parquets..."
for f in "$DATA_DIR"/*.parquet; do
    rsync -az "$f" root@"$SERVER_IP":/root/neat/data/indicators/ 2>/dev/null
done
PARQUET_COUNT=$(ssh root@"$SERVER_IP" 'ls /root/neat/data/indicators/*.parquet | wc -l')
log "Uploaded $PARQUET_COUNT indicator parquets"

# ── 7. Transfer code ─────────────────────────────────────────────
log "Transferring code..."
rsync -az "$SCRIPT_DIR/export_swing_indicators.py"  root@"$SERVER_IP":/root/neat/
rsync -az "$REPO_DIR/lib/indicators.py"              root@"$SERVER_IP":/root/neat/lib/
rsync -az "$REPO_DIR/lib/swing_indicators.py"        root@"$SERVER_IP":/root/neat/lib/
rsync -az "$REPO_DIR/lib/asi_indicator.py"           root@"$SERVER_IP":/root/neat/lib/
rsync -az "$REPO_DIR/lib/pair_config.py"             root@"$SERVER_IP":/root/neat/lib/ 2>/dev/null || true
ssh root@"$SERVER_IP" 'touch /root/neat/lib/__init__.py'
log "Code transferred"

# ── 8. Run export ────────────────────────────────────────────────
log "Running export_swing_indicators.py..."
log "(~15-20 min for 12 pairs, includes IC pre-screening report)"
ssh -o StrictHostKeyChecking=no root@"$SERVER_IP" "
    source /root/venv/bin/activate &&
    cd /root/neat &&
    S5_DATA_DIR=/root/neat/data/s5 \
    ASI_MC_DATA_DIR=/root/neat/data/indicators \
    PYTHONUNBUFFERED=1 \
    python3 export_swing_indicators.py 2>&1
" | tee -a "$LOG"

# Verify all columns were added
VERIFY=$(ssh root@"$SERVER_IP" "
    source /root/venv/bin/activate &&
    python3 - <<'PYEOF'
import pandas as pd
from pathlib import Path
DATA = Path('/root/neat/data/indicators')
NEW_COLS = ['sb_a','erp_a','hh_asi','hl_asi','erp_p','hh_price','hl_price','d_erp_p','d_erp_a']
ok = 0
for f in sorted(DATA.glob('*_asi_mc.parquet')):
    df = pd.read_parquet(f, engine='pyarrow')
    missing = [c for c in NEW_COLS if c not in df.columns]
    if missing:
        print(f'MISSING {f.stem}: {missing}')
    else:
        ok += 1
print(f'{ok}/12 parquets have all swing columns')
PYEOF
")
log "$VERIFY"
if ! echo "$VERIFY" | grep -q "12/12"; then
    log "ERROR: Not all parquets have swing columns. Check log."
    exit 1
fi

# ── 9. Copy augmented parquets back ──────────────────────────────
log "Downloading augmented parquets → local data/asi_mc_indicators/..."
for pair in AUD_JPY AUD_USD CAD_JPY CHF_JPY EUR_GBP EUR_JPY EUR_USD GBP_JPY GBP_USD NZD_JPY NZD_USD USD_JPY; do
    scp -o StrictHostKeyChecking=no \
        root@"$SERVER_IP":/root/neat/data/indicators/${pair}_asi_mc.parquet \
        "$DATA_DIR/${pair}_asi_mc.parquet" &&
        log "  $pair: OK"
done
log "All parquets updated locally"

# ── 10. Verify local parquets ─────────────────────────────────────
python3 - <<'PYEOF'
import pandas as pd
from pathlib import Path
DATA = Path("data/asi_mc_indicators")
NEW_COLS = ['sb_a','erp_a','hh_asi','hl_asi','erp_p','hh_price','hl_price','d_erp_p','d_erp_a']
print("\nLocal parquet verification:")
for f in sorted(DATA.glob("*_asi_mc.parquet")):
    df = pd.read_parquet(f, engine='pyarrow')
    has_swing = all(c in df.columns for c in NEW_COLS)
    status = "OK" if has_swing else f"MISSING: {[c for c in NEW_COLS if c not in df.columns]}"
    print(f"  {f.stem.replace('_asi_mc',''):<10}: {len(df):>7,} bars  {len(df.columns)} cols  {status}")
PYEOF

# ── 11. Detach volume + delete server ────────────────────────────
log "Detaching volume..."
hcloud volume detach "$VOLUME_NAME" && log "Volume detached"
log "Deleting server..."
hcloud server delete swingdim-export --yes && log "Server deleted"

log ""
log "═══ EXPORT COMPLETE ═══"
log "Local parquets now have all 9 swing columns."
log ""
log "Next step — launch training:"
log "  bash research/experiments/asi_mc/deploy_swingdim.sh"
