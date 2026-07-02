#!/bin/bash
# Deploy slot-4 CMA grid to Hetzner cx53 servers.
# Usage: deploy_slot4.sh <candidate> [N_SERVERS=4]
#
# Assumes:
#   - hcloud CLI authenticated
#   - SSH key "user@host" registered with Hetzner
#   - neat-data volume (hel1) exists with M5 OHLC + smoother-causal parquets
#   - Candidate already ported + validated locally + pushed to GitHub
#
# Splits 12 pairs × 4 seeds = 48 runs across N_SERVERS.
# Each server: ~12 runs × 200 gens × 4 pop-workers = ~10-15 min.
set -euo pipefail

CAND="${1:?candidate name required}"
N=${2:-4}
LOCATION="hel1"
TYPE="cx53"
SSH_KEY="user@host"
VOLUME="neat-data"
REPO="https://github.com/<GITHUB_USER>/fx-core.git"

PAIRS=(EUR_USD GBP_USD USD_JPY AUD_USD EUR_JPY GBP_JPY AUD_JPY CAD_JPY CHF_JPY NZD_JPY NZD_USD EUR_GBP)
SEEDS=(42 137 23 7)

echo "=== Deploying slot-4 grid for candidate '$CAND' on $N servers ==="

# 1. Provision
SERVERS=()
for i in $(seq 1 $N); do
    NAME="slot4-$CAND-$i"
    hcloud server create --name "$NAME" --type "$TYPE" --image ubuntu-24.04 \
        --location "$LOCATION" --ssh-key "$SSH_KEY" > /dev/null
    IP=$(hcloud server ip "$NAME")
    SERVERS+=("$NAME:$IP")
    echo "  [$i/$N] $NAME $IP"
done

# 2. Attach volume to server 1 for parquet sync
echo "=== Distributing data ==="
FIRST_NAME=$(echo "${SERVERS[0]}" | cut -d: -f1)
FIRST_IP=$(echo "${SERVERS[0]}" | cut -d: -f2)
hcloud volume attach "$VOLUME" --server "$FIRST_NAME"
sleep 10  # let volume mount register
ssh -o StrictHostKeyChecking=no "root@$FIRST_IP" "
    set -e
    apt-get -qq update && apt-get -qq install -y git python3-pip
    mkdir -p /mnt/neat-data && mount /dev/disk/by-id/scsi-0HC_Volume_105213043 /mnt/neat-data
    git clone $REPO /root/fx-core
    cd /root/fx-core && pip install -q cma numpy pandas numba pyarrow python-dotenv requests cmaes
    mkdir -p /root/fx-core/data/m5_ohlc
    cp /mnt/neat-data/*_M5.parquet /root/fx-core/data/m5_ohlc/
    cd /root/fx-core
    python3 research/experiments/cma_5in/build_causal_parquets.py --smoother kalman10 --workers 8
"

# 3. Distribute parquets to other servers (internal network)
for s in "${SERVERS[@]:1}"; do
    IP=$(echo "$s" | cut -d: -f2)
    echo "  rsync -> $IP"
    ssh "root@$FIRST_IP" "rsync -az /root/fx-core/data/m5_ohlc/ root@$IP:/root/fx-core/data/m5_ohlc/ && rsync -az /root/fx-core/ root@$IP:/root/fx-core/"
done

hcloud volume detach "$VOLUME"

# 4. Shard runs across servers
echo "=== Launching CMA grid ==="
N_RUNS=$((${#PAIRS[@]} * ${#SEEDS[@]}))
IDX=0
for pair in "${PAIRS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        S_IDX=$((IDX % N))
        S_IP=$(echo "${SERVERS[$S_IDX]}" | cut -d: -f2)
        ssh "root@$S_IP" "
            cd /root/fx-core && mkdir -p research/experiments/cma_5in/indicator_loop/results
            nohup python3 research/experiments/cma_5in/indicator_loop/test_slot4_swap_cma.py \
                --candidate $CAND --pair $pair --seed $seed \
                --gens 200 --pop 40 --workers 4 \
                > research/experiments/cma_5in/indicator_loop/results/slot4_${CAND}_${pair}_s${seed}.log 2>&1 &
        "
        echo "  [$((IDX+1))/$N_RUNS] $CAND $pair s$seed -> server $S_IDX ($S_IP)"
        IDX=$((IDX+1))
    done
done

echo
echo "=== Launched. Monitor: ==="
for s in "${SERVERS[@]}"; do
    NAME=$(echo "$s" | cut -d: -f1)
    IP=$(echo "$s" | cut -d: -f2)
    echo "  ssh root@$IP 'ls /root/fx-core/research/experiments/cma_5in/indicator_loop/results/*.json | wc -l'"
done
echo
echo "When done: bash collect_slot4.sh $CAND ${SERVERS[*]}"
