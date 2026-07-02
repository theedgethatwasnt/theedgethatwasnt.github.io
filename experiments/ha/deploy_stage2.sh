#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# HA Experiment Stage 2 — Deploy to 4 Hetzner servers
#
# 4 runs total (2 variants × 2 seeds):
#   Server 1: S2-long  seed 42
#   Server 2: S2-long  seed 137
#   Server 3: S2-both  seed 42
#   Server 4: S2-both  seed 137
#
# Each server: cx53 (16 vCPU, 32GB), ~$0.10/hr, ~2.5 hrs = ~$1 total
# ═══════════════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "═══════════════════════════════════════════════════════"
echo "HA Stage 2: Full Training Deploy"
echo "═══════════════════════════════════════════════════════"

# ── Phase 1: Create 4 servers ──
echo "Creating 4 servers..."
SERVERS=()
for i in 1 2 3 4; do
    NAME="ha-stage2-$i"
    echo "  Creating $NAME..."
    IP=$(hcloud server create --name "$NAME" --type cx53 --image ubuntu-24.04 \
        --location hel1 --ssh-key "user@host" \
        --format json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['server']['public_net']['ipv4']['ip'])" 2>/dev/null)
    if [ -z "$IP" ]; then
        echo "  hel1 failed, trying nbg1..."
        IP=$(hcloud server create --name "$NAME" --type cx53 --image ubuntu-24.04 \
            --location nbg1 --ssh-key "user@host" \
            --format json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['server']['public_net']['ipv4']['ip'])" 2>/dev/null)
    fi
    echo "  $NAME: $IP"
    SERVERS+=("$IP")
done

echo ""
echo "Servers: ${SERVERS[*]}"
echo ""

# Save IPs for later use
echo "${SERVERS[*]}" > "$SCRIPT_DIR/stage2_servers.txt"

# ── Phase 2: Setup primary + upload data ──
PRIMARY="${SERVERS[0]}"
echo "Setting up primary server ($PRIMARY)..."
echo "  Installing packages..."
ssh -o StrictHostKeyChecking=no root@$PRIMARY \
    'apt-get update -qq && apt-get install -y -qq python3-pip python3-venv > /dev/null 2>&1 && python3 -m venv /root/venv && source /root/venv/bin/activate && pip install -q neat-python numba pandas pyarrow numpy && mkdir -p /root/neat/data /root/neat/results/stage2'

echo "  Uploading parquet data..."
rsync -az --progress "$REPO_DIR/data/scalper_parquet/"*_S5_BA.parquet root@$PRIMARY:/root/neat/data/

echo "  Uploading training scripts..."
rsync -az "$SCRIPT_DIR/stage2_training.py" root@$PRIMARY:/root/neat/
rsync -az "$SCRIPT_DIR/neat_config_2out.ini" root@$PRIMARY:/root/neat/
rsync -az "$SCRIPT_DIR/neat_config_3out.ini" root@$PRIMARY:/root/neat/
rsync -az "$REPO_DIR/lib/fast_eval.py" root@$PRIMARY:/root/neat/lib/
rsync -az "$REPO_DIR/lib/pair_config.py" root@$PRIMARY:/root/neat/lib/

# Create __init__.py for lib module
ssh root@$PRIMARY 'touch /root/neat/lib/__init__.py'

# ── Phase 3: Distribute to other servers ──
echo ""
echo "Setting up other servers and distributing data..."
ssh root@$PRIMARY "echo 'StrictHostKeyChecking no' >> /root/.ssh/config && ssh-keygen -t ed25519 -f /root/.ssh/id_ed25519 -N '' -q 2>/dev/null || true"
PUBKEY=$(ssh root@$PRIMARY "cat /root/.ssh/id_ed25519.pub")

for IP in "${SERVERS[@]:1}"; do
    ssh -o StrictHostKeyChecking=no root@$IP \
        "apt-get update -qq && apt-get install -y -qq python3-pip python3-venv > /dev/null 2>&1 && python3 -m venv /root/venv && source /root/venv/bin/activate && pip install -q neat-python numba pandas pyarrow numpy && mkdir -p /root/neat/data /root/neat/results/stage2 /root/neat/lib && echo '$PUBKEY' >> /root/.ssh/authorized_keys" &
done
wait

echo "  Syncing data from primary to others..."
ssh root@$PRIMARY "for IP in ${SERVERS[1]} ${SERVERS[2]} ${SERVERS[3]}; do rsync -az -e 'ssh -o StrictHostKeyChecking=no' /root/neat/ root@\$IP:/root/neat/ & done; wait; echo 'Distribution complete'"

# ── Phase 4: Launch training ──
echo ""
echo "Launching training runs..."

VARIANTS=("S2-long" "S2-long" "S2-both" "S2-both")
SEEDS=(42 137 42 137)

for i in 0 1 2 3; do
    IP="${SERVERS[$i]}"
    VAR="${VARIANTS[$i]}"
    SEED="${SEEDS[$i]}"
    LOG="/root/neat/results/stage2/${VAR}_s${SEED}.log"
    echo "  Server $((i+1)) ($IP): $VAR seed $SEED"
    ssh root@$IP "source /root/venv/bin/activate && cd /root/neat && \
        NEAT_DATA_DIR=/root/neat/data PYTHONUNBUFFERED=1 \
        nohup python3 stage2_training.py --variant $VAR --seed $SEED \
            --generations 200 --pop-size 150 \
        > $LOG 2>&1 &"
done

echo ""
echo "═══════════════════════════════════════════════════════"
echo "All 4 training runs launched!"
echo ""
echo "Monitor:"
for i in 0 1 2 3; do
    echo "  ssh root@${SERVERS[$i]} 'tail -f /root/neat/results/stage2/${VARIANTS[$i]}_s${SEEDS[$i]}.log'"
done
echo ""
echo "Collect results when done:"
echo "  $SCRIPT_DIR/collect_stage2.sh"
echo ""
echo "Cleanup:"
echo "  for i in 1 2 3 4; do hcloud server delete ha-stage2-\$i --yes; done"
echo "═══════════════════════════════════════════════════════"
