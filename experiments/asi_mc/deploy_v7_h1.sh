#!/bin/bash
# IronNet V7 H1 — 12 pairs on 3 Hetzner servers
set -e

GENS=200; PRETRAIN_GENS=50; SINE_GENS=30; POP=150; ISLANDS=4; STALL=60; SEED=42

declare -a SP
SP[1]="EUR_JPY USD_JPY GBP_JPY AUD_JPY"
SP[2]="CAD_JPY CHF_JPY NZD_JPY EUR_USD"
SP[3]="GBP_USD AUD_USD NZD_USD EUR_GBP"

PD="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../.." && pwd )"
SF="$PD/research/experiments/asi_mc/v7_servers.txt"
mkdir -p "$PD/research/experiments/asi_mc/results/ironnet_v7"

echo "Creating 3 ccx23 servers..."
> "$SF"
for i in 1 2 3; do
    hcloud server create --name "v7-h1-$i" --type ccx23 --image ubuntu-24.04 \
        --location hel1 --ssh-key "user@host" 2>/dev/null || \
    hcloud server create --name "v7-h1-$i" --type cx43 --image ubuntu-24.04 \
        --location hel1 --ssh-key "user@host" 2>/dev/null
    IP=$(hcloud server ip "v7-h1-$i")
    echo "$i $IP" >> "$SF"; echo "  v7-h1-$i → $IP"
done

echo "Waiting 30s..."; sleep 30

while read IDX IP; do
    echo "Setting up server $IDX ($IP)..."
    ssh-keygen -f "$HOME/.ssh/known_hosts" -R "$IP" 2>/dev/null || true
    ssh -o StrictHostKeyChecking=no root@$IP bash -s <<'SETUP'
apt-get update -qq && apt-get install -y -qq python3-pip python3-venv > /dev/null 2>&1
python3 -m venv /root/venv
source /root/venv/bin/activate
pip install -q numpy pandas numba neat-python pyarrow
mkdir -p /root/fx-core/lib /root/fx-core/research/experiments/asi_mc/results/ironnet_v7 /root/fx-core/data/v7_indicators
SETUP
    scp -q "$PD/lib/fast_eval.py" "$PD/lib/asi_indicator.py" root@$IP:/root/fx-core/lib/
    scp -q "$PD/research/experiments/asi_mc/train_ironnet_perpair.py" root@$IP:/root/fx-core/research/experiments/asi_mc/
    scp -q "$PD/research/experiments/asi_mc/export_v7_training_data.py" root@$IP:/root/fx-core/research/experiments/asi_mc/
    scp -q "$PD/research/experiments/asi_mc/neat_config_7in_3out.ini" root@$IP:/root/fx-core/research/experiments/asi_mc/
    scp -q "$PD/research/experiments/asi_mc/neat_config_4in_3out.ini" root@$IP:/root/fx-core/research/experiments/asi_mc/
    rsync -az "$PD/data/v7_indicators/" root@$IP:/root/fx-core/data/v7_indicators/
    echo "  Done"
done < "$SF"

echo "Launching training..."
while read IDX IP; do
    PAIRS="${SP[$IDX]}"
    echo "  Server $IDX ($IP): $PAIRS"
    ssh root@$IP "nohup bash -c '
source /root/venv/bin/activate
cd /root/fx-core/research/experiments/asi_mc
for PAIR in $PAIRS; do
    echo \"[\$(date)] Starting \$PAIR...\"
    PYTHONUNBUFFERED=1 python3 train_ironnet_perpair.py \
        --pair \$PAIR --seed $SEED --tf H1 --mode v7 \
        --gens $GENS --pretrain-gens $PRETRAIN_GENS --sine-gens $SINE_GENS \
        --pop $POP --islands $ISLANDS --stall-limit $STALL \
        > results/ironnet_v7/\${PAIR}_s${SEED}.log 2>&1 || true
    echo \"[\$(date)] Done \$PAIR\"
done
echo \"[\$(date)] ALL DONE\"
' > /root/training.log 2>&1 &"
done < "$SF"

echo ""
echo "All launched! Monitor:"
while read IDX IP; do
    echo "  ssh root@$IP 'tail -f /root/fx-core/research/experiments/asi_mc/results/ironnet_v7/*.log'"
done < "$SF"
