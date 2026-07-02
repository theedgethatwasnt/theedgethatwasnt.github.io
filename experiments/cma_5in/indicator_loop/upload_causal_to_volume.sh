#!/bin/bash
# Upload locally-built 43-feature causal parquets to the neat-data Hetzner volume.
# Uses a temp cx11 server to mount the volume, scp parquets up, move to volume, detach, delete.
#
# Usage: upload_causal_to_volume.sh
#
# Requires: all 12 pairs' {PAIR}_M5_kalman10_causal.parquet in data/m5_ohlc/
set -euo pipefail

PAIRS=(EUR_USD GBP_USD USD_JPY AUD_USD EUR_JPY GBP_JPY AUD_JPY CAD_JPY CHF_JPY NZD_JPY NZD_USD EUR_GBP)
VOLUME="neat-data"
LOCATION="hel1"
SSH_KEY="user@host"
TEMP_NAME="temp-upload-$(date +%s)"

cd "$(dirname "$0")/../../../.."

# 1. Verify all 12 parquets exist locally
echo "=== Verifying local parquets ==="
missing=0
for p in "${PAIRS[@]}"; do
    f="data/m5_ohlc/${p}_M5_kalman10_causal.parquet"
    if [ ! -f "$f" ]; then
        echo "  ❌ $f MISSING"
        missing=1
    else
        size=$(stat -c%s "$f")
        echo "  ✓ $p: $(numfmt --to=iec $size)"
    fi
done
[ "$missing" = "1" ] && { echo "Build missing parquets first."; exit 1; }

# 2. Create temp cx11
echo ""
echo "=== Creating temp cx11 ($TEMP_NAME) ==="
hcloud server create --name "$TEMP_NAME" --type cx23 --image ubuntu-24.04 \
    --location "$LOCATION" --ssh-key "$SSH_KEY" > /dev/null
IP=$(hcloud server ip "$TEMP_NAME")
echo "  $TEMP_NAME at $IP"
sleep 15  # wait for SSH

# 3. Attach volume
hcloud volume attach "$VOLUME" --server "$TEMP_NAME"
sleep 10

# 4. Mount volume + create upload dir
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 "root@$IP" "
    mkdir -p /mnt/neat-data
    mount /dev/disk/by-id/scsi-0HC_Volume_105213043 /mnt/neat-data
    mkdir -p /mnt/neat-data/
    echo 'Volume mounted:'
    df -h /mnt/neat-data
    echo 'Current contents:'
    ls /mnt/neat-data/*_kalman10_causal.parquet 2>/dev/null || echo '  (no causal parquets yet)'
"

# 5. Rsync parquets from local to volume (directly)
echo ""
echo "=== Uploading 12 parquets to volume ==="
rsync -avz --progress \
    -e "ssh -o StrictHostKeyChecking=no" \
    data/m5_ohlc/*_kalman10_causal.parquet \
    "root@$IP:/mnt/neat-data/"

# 6. Verify
echo ""
echo "=== Verify on volume ==="
ssh "root@$IP" "ls -lh /mnt/neat-data/*_kalman10_causal.parquet | awk '{print \$5, \$9}'"

# 7. Unmount + detach + delete temp
ssh "root@$IP" "sync; umount /mnt/neat-data"
hcloud volume detach "$VOLUME"
hcloud server delete "$TEMP_NAME" --yes

echo ""
echo "=== Upload complete. 12 causal parquets now on neat-data volume. ==="
python3 -c "
import sys; sys.path.insert(0,'.')
from lib.notify import _send
_send('📤 Uploaded 12 × 43-feature causal parquets to neat-data volume (~1.2 GB). Future tier-2 deploys skip rebuild.')
"
