#!/bin/bash
# Feature Set Experiments A-F — parallel on single server, single pair
# Architecture: N→7→4→3 (2 hidden layers), all wavelet activations
# Training: sine 20g → zigzag 30g → WF P&L 150g, 4 islands × 150 pop
set -e

PAIR="${1:-EUR_GBP}"
GENS=150; PT=30; SINE=20; POP=150; ISLANDS=4; STALL=40; SEED=42
SETS="setA setB setC setD setE setF"

PD="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../.." && pwd )"

echo "Creating ccx33 server (8 vCPU)..."
hcloud server create --name "sets-af" --type ccx33 --image ubuntu-24.04 \
    --location hel1 --ssh-key "user@host" 2>/dev/null || \
hcloud server create --name "sets-af" --type cx43 --image ubuntu-24.04 \
    --location hel1 --ssh-key "user@host" 2>/dev/null
IP=$(hcloud server ip "sets-af")
echo "Server: $IP"
echo "$IP" > "$PD/research/experiments/asi_mc/sets_af_server.txt"

sleep 25
ssh-keygen -f "$HOME/.ssh/known_hosts" -R "$IP" 2>/dev/null || true

echo "Setting up..."
ssh -o StrictHostKeyChecking=no root@$IP bash -s <<'SETUP'
apt-get update -qq && apt-get install -y -qq python3-pip python3-venv > /dev/null 2>&1
python3 -m venv /root/venv
source /root/venv/bin/activate
pip install -q numpy pandas numba neat-python pyarrow
mkdir -p /root/fx-core/lib /root/fx-core/data/unified_indicators
for s in setA setB setC setD setE setF; do
    mkdir -p /root/fx-core/research/experiments/asi_mc/results/ironnet_$s
done
SETUP

scp -q "$PD/lib/fast_eval.py" "$PD/lib/asi_indicator.py" "$PD/lib/swing_indicators.py" root@$IP:/root/fx-core/lib/
scp -q "$PD/research/experiments/asi_mc/train_ironnet_perpair.py" root@$IP:/root/fx-core/research/experiments/asi_mc/
scp -q "$PD/research/experiments/asi_mc/export_unified_training_data.py" root@$IP:/root/fx-core/research/experiments/asi_mc/
scp -q "$PD/research/experiments/asi_mc/neat_config_4in_3out.ini" root@$IP:/root/fx-core/research/experiments/asi_mc/
scp -q "$PD/research/experiments/asi_mc/neat_config_5in_3out.ini" root@$IP:/root/fx-core/research/experiments/asi_mc/
rsync -az "$PD/data/unified_indicators/" root@$IP:/root/fx-core/data/unified_indicators/

echo "Launching 6 sets in parallel on $PAIR..."
for SET in $SETS; do
    ssh root@$IP "nohup bash -c '
source /root/venv/bin/activate
cd /root/fx-core/research/experiments/asi_mc
PYTHONUNBUFFERED=1 python3 train_ironnet_perpair.py \
    --pair $PAIR --seed $SEED --tf H1 --mode $SET \
    --gens $GENS --pretrain-gens $PT --sine-gens $SINE \
    --pop $POP --islands $ISLANDS --stall-limit $STALL \
    > results/ironnet_${SET}/${PAIR}_s${SEED}.log 2>&1 || true
echo DONE_${SET}
' > /root/${SET}.log 2>&1 &"
    echo "  $SET launched"
done

echo ""
echo "All 6 sets running on $IP for $PAIR"
echo "Monitor: ssh root@$IP 'tail -f /root/fx-core/research/experiments/asi_mc/results/ironnet_set*/*.log'"
echo "Collect: for s in $SETS; do scp root@$IP:/root/fx-core/research/experiments/asi_mc/results/ironnet_\$s/*result*.json .; done"
