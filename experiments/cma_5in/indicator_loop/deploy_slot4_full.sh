#!/bin/bash
# Full tier-2 deploy: ALL candidates × 12 pairs × 4 seeds × 200 gens, sharded across N Hetzner cx53.
#
# Usage: deploy_slot4_full.sh [N_SERVERS=10]
#
# Shards 2064 jobs across N servers.
# Each server runs 4 concurrent CMAs (workers=4 each = saturates 16 vCPU).
# Wall time ≈ 2064 / (N × 4) × 3.5 min. At N=10 → ~3 h.
# Cost ≈ N × €0.10/h × hours.

set -euo pipefail

N=${1:-10}
LOCATION="hel1"
TYPE="cx53"
SSH_KEY="user@host"
VOLUME="neat-data"
REPO="https://github.com/<GITHUB_USER>/fx-core.git"

PAIRS=(EUR_USD GBP_USD USD_JPY AUD_USD EUR_JPY GBP_JPY AUD_JPY CAD_JPY CHF_JPY NZD_JPY NZD_USD EUR_GBP)
SEEDS=(42 137 23 7)
CANDS=$(python3 -c "
import json
d = json.load(open('research/experiments/cma_5in/indicator_loop/candidates.json'))
out = [c['name'] for c in d['candidates'] if c.get('port_status') == 'ported']
print(' '.join(out))
")
CAND_ARR=($CANDS)
N_CANDS=${#CAND_ARR[@]}
N_JOBS=$((N_CANDS * ${#PAIRS[@]} * ${#SEEDS[@]}))

MSG() {
    python3 -c "import sys; sys.path.insert(0,'.'); from lib.notify import _send; _send('''$1''')"
}

echo "=== Full tier-2 deploy ==="
echo "Candidates: $N_CANDS"
echo "Pairs: ${#PAIRS[@]}, Seeds: ${#SEEDS[@]}"
echo "Total jobs: $N_JOBS"
echo "Servers: $N"
echo "Est. wall time: $(echo "$N_JOBS / ($N * 4) * 3.5" | bc) min"
MSG "🚀 Tier-2 full deploy starting: $N_CANDS cands × 12 pairs × 4 seeds × 200g = $N_JOBS jobs on $N Hetzner cx53"

# ── 1. Provision servers ──────────────────────────────────────
SERVERS=()
for i in $(seq 1 $N); do
    NAME="slot4-tier2-$i"
    hcloud server create --name "$NAME" --type "$TYPE" --image ubuntu-24.04 \
        --location "$LOCATION" --ssh-key "$SSH_KEY" > /dev/null 2>&1 || {
        echo "Failed to create $NAME"
        exit 1
    }
    IP=$(hcloud server ip "$NAME")
    SERVERS+=("$NAME:$IP")
    echo "  [$i/$N] $NAME $IP"
done
MSG "🖥 Provisioned $N servers"

# ── 2. Attach volume + rebuild data on server 1 ───────────────
FIRST_NAME=$(echo "${SERVERS[0]}" | cut -d: -f1)
FIRST_IP=$(echo "${SERVERS[0]}" | cut -d: -f2)
echo "=== Provisioning server 1 with neat-data ==="
hcloud volume attach "$VOLUME" --server "$FIRST_NAME"
sleep 15

ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 "root@$FIRST_IP" bash <<EOF
set -e
apt-get -qq update
apt-get -qq install -y git python3-pip rsync
mkdir -p /mnt/neat-data
mount /dev/disk/by-id/scsi-0HC_Volume_105213043 /mnt/neat-data 2>/dev/null || true
git clone $REPO /root/fx-core
cd /root/fx-core && pip install -q cma numpy pandas numba pyarrow python-dotenv requests cmaes
mkdir -p /root/fx-core/data/m5_ohlc
# Use pre-built causal parquets from volume (single source of truth, post-RCA, locally validated)
echo "Pulling pre-built causal parquets from neat-data volume..."
cp /mnt/neat-data/*_M5_kalman10_causal.parquet /root/fx-core/data/m5_ohlc/
count=\$(ls /root/fx-core/data/m5_ohlc/*_M5_kalman10_causal.parquet 2>/dev/null | wc -l)
echo "Causal parquets on disk: \$count"
if [ "\$count" -ne "12" ]; then
    echo "ERROR: expected 12 causal parquets, got \$count. Run upload_causal_to_volume.sh first."
    exit 1
fi
ls -lh /root/fx-core/data/m5_ohlc/*_kalman10_causal.parquet | head -3
EOF
MSG "📦 Server 1 provisioned + 12 pre-built causal parquets pulled from volume"

# ── 3. Distribute code + parquets to other servers in parallel ────
echo "=== Distributing to other servers ==="
for s in "${SERVERS[@]:1}"; do
    IP=$(echo "$s" | cut -d: -f2)
    (
        ssh -o StrictHostKeyChecking=no "root@$IP" "
            apt-get -qq update && apt-get -qq install -y git python3-pip rsync > /dev/null
            pip install -q cma numpy pandas numba pyarrow python-dotenv requests cmaes
            mkdir -p /root/fx-core
        " 2>&1 | head -3
        ssh "root@$FIRST_IP" "rsync -az --exclude='.git' /root/fx-core/ root@$IP:/root/fx-core/"
        echo "  ✓ $IP ready"
    ) &
done
wait
echo "All servers provisioned."
hcloud volume detach "$VOLUME"
MSG "✅ All $N servers ready. Launching job queue..."

# ── 4. Shard jobs and launch on each server ───────────────────
# Build full job list
> /tmp/slot4_jobs.txt
for cand in "${CAND_ARR[@]}"; do
    for pair in "${PAIRS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "$cand $pair $seed" >> /tmp/slot4_jobs.txt
        done
    done
done
echo "Total jobs queued: $(wc -l < /tmp/slot4_jobs.txt)"

# Shard: job i goes to server (i % N)
for i in $(seq 0 $((N-1))); do
    awk -v n=$N -v idx=$i 'NR % n == idx' /tmp/slot4_jobs.txt > /tmp/slot4_jobs_$i.txt
    echo "Server $i: $(wc -l < /tmp/slot4_jobs_$i.txt) jobs"
done

# Copy job list to each server, launch runner
for i in $(seq 0 $((N-1))); do
    IP=$(echo "${SERVERS[$i]}" | cut -d: -f2)
    scp /tmp/slot4_jobs_$i.txt "root@$IP:/tmp/jobs.txt" > /dev/null 2>&1

    # Runner script on server: runs each job, up to CONCURRENCY at a time
    ssh "root@$IP" "cat > /root/run_jobs.sh" <<'EOF_RUNNER'
#!/bin/bash
cd /root/fx-core
mkdir -p research/experiments/cma_5in/indicator_loop/results
CONCURRENCY=4
pids=()
while IFS=" " read -r cand pair seed; do
    # Throttle concurrency
    while [ ${#pids[@]} -ge $CONCURRENCY ]; do
        new_pids=()
        for p in "${pids[@]}"; do
            if kill -0 $p 2>/dev/null; then
                new_pids+=($p)
            fi
        done
        pids=("${new_pids[@]}")
        [ ${#pids[@]} -ge $CONCURRENCY ] && sleep 5
    done
    log="research/experiments/cma_5in/indicator_loop/results/slot4_${cand}_${pair}_s${seed}.log"
    nohup python3 research/experiments/cma_5in/indicator_loop/test_slot4_swap_cma.py \
        --candidate "$cand" --pair "$pair" --seed "$seed" \
        --gens 200 --pop 40 --workers 4 > "$log" 2>&1 &
    pids+=($!)
done < /tmp/jobs.txt
wait
echo "All jobs done on $(hostname)"
EOF_RUNNER
    ssh "root@$IP" "chmod +x /root/run_jobs.sh && nohup /root/run_jobs.sh > /root/run_jobs.log 2>&1 &"
    echo "  Server $i launched"
done

MSG "⚙ Jobs launched on all servers. Monitor with collect_slot4_full.sh or ssh into servers."

# Write server list for collect script
echo "=== Server list (save for collect) ==="
for s in "${SERVERS[@]}"; do
    echo "$s"
done > research/experiments/cma_5in/indicator_loop/tier2_servers.txt
cat research/experiments/cma_5in/indicator_loop/tier2_servers.txt

echo
echo "=== Deploy complete. ==="
echo "Monitor: for ip in \$(cut -d: -f2 tier2_servers.txt); do ssh root@\$ip 'ls /root/fx-core/research/experiments/cma_5in/indicator_loop/results/*.json 2>/dev/null | wc -l'; done"
echo "Collect: bash collect_slot4_full.sh"
