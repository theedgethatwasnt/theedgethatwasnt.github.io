#!/bin/bash
# Resume v2: creates missing 5-10, provisions ALL from local (no git clone,
# no server-to-server rsync — everything scp'd from local in parallel).
set -uo pipefail

cd "$(dirname "$0")/../../../.."

LOCATION="hel1"
TYPE="cx53"
SSH_KEY="user@host"
VOLUME="neat-data"

PAIRS=(EUR_USD GBP_USD USD_JPY AUD_USD EUR_JPY GBP_JPY AUD_JPY CAD_JPY CHF_JPY NZD_JPY NZD_USD EUR_GBP)
SEEDS=(42 137 23 7)
CANDS=$(python3 -c "
import json
d = json.load(open('research/experiments/cma_5in/indicator_loop/candidates.json'))
print(' '.join([c['name'] for c in d['candidates'] if c.get('port_status') == 'ported']))
")
CAND_ARR=($CANDS)

MSG() {
    python3 -c "import sys; sys.path.insert(0,'.'); from lib.notify import _send; _send('''$1''')"
}

# 1. Create missing servers 5-10 with retry
echo "=== Creating missing servers 5-10 ==="
for i in 5 6 7 8 9 10; do
    NAME="slot4-tier2-$i"
    if hcloud server describe "$NAME" >/dev/null 2>&1; then
        IP=$(hcloud server ip "$NAME")
        echo "  $NAME already exists at $IP"
        continue
    fi
    for attempt in 1 2 3; do
        if hcloud server create --name "$NAME" --type "$TYPE" --image ubuntu-24.04 \
             --location "$LOCATION" --ssh-key "$SSH_KEY" >/dev/null 2>&1; then
            IP=$(hcloud server ip "$NAME")
            echo "  ✓ [$i/10] $NAME $IP (attempt $attempt)"
            break
        else
            echo "  ✗ [$i/10] $NAME attempt $attempt failed"
            sleep 8
        fi
    done
done

# 2. Gather all online
SERVERS=()
for i in 1 2 3 4 5 6 7 8 9 10; do
    NAME="slot4-tier2-$i"
    if hcloud server describe "$NAME" >/dev/null 2>&1; then
        IP=$(hcloud server ip "$NAME")
        SERVERS+=("$NAME:$IP")
    fi
done
N=${#SERVERS[@]}
echo ""
echo "=== $N/10 servers online ==="
for s in "${SERVERS[@]}"; do echo "  $s"; done
MSG "🖥 Tier-2: $N/10 servers online, provisioning via scp from local..."

# 3. Wait for SSH on all servers (fresh servers take ~30-60s)
echo ""
echo "=== Waiting for SSH ==="
for s in "${SERVERS[@]}"; do
    IP=$(echo "$s" | cut -d: -f2)
    for attempt in 1 2 3 4 5 6 7 8; do
        if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes \
              "root@$IP" "echo ok" >/dev/null 2>&1; then
            echo "  ✓ $IP ssh ok"
            break
        fi
        sleep 5
    done
done

# 4. Attach volume to server 1, copy parquets to its disk (but we'll scp locally too to avoid fragility)
FIRST_NAME=$(echo "${SERVERS[0]}" | cut -d: -f1)
FIRST_IP=$(echo "${SERVERS[0]}" | cut -d: -f2)
hcloud volume attach "$VOLUME" --server "$FIRST_NAME" 2>&1 | tail -1
sleep 15

# 5. PROVISION ALL SERVERS IN PARALLEL FROM LOCAL ──────────────
# Each: apt install -> pip install -> scp code + parquets -> done
LOCAL_FX=/path/to/projects/fx-core

echo ""
echo "=== Provisioning all $N servers from local (parallel) ==="

# First: tar up the code (exclude .git, data that we copy separately, results dirs)
cd "$LOCAL_FX"
tar czf /tmp/fxcore_code.tgz \
    --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='data/m5_ohlc/*.parquet' \
    --exclude='research/experiments/cma_5in/indicator_loop/results' \
    --exclude='research/experiments/cma_5in/indicator_loop/results_tier2' \
    --exclude='experiments/*/results' \
    --exclude='experiments/*/data' \
    lib/ services/ research/experiments/cma_5in/ data/m5_ohlc/ 2>/dev/null || {
      # Fallback: no data — we'll scp parquets separately
      tar czf /tmp/fxcore_code.tgz \
          --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
          lib/ research/experiments/cma_5in/ 2>/dev/null
}
tar czf /tmp/fxcore_parquets.tgz data/m5_ohlc/*_kalman10_causal.parquet 2>/dev/null
echo "Code: $(du -h /tmp/fxcore_code.tgz | cut -f1), Parquets: $(du -h /tmp/fxcore_parquets.tgz | cut -f1)"

provision_one() {
    local IP=$1
    ssh -o StrictHostKeyChecking=no "root@$IP" \
        "apt-get -qq update >/dev/null 2>&1 && \
         apt-get -qq install -y python3-pip rsync >/dev/null 2>&1 && \
         pip install -q cma numpy pandas numba pyarrow python-dotenv requests cmaes && \
         mkdir -p /root/fx-core/data/m5_ohlc /root/fx-core/research/experiments/cma_5in/indicator_loop/results" 2>/dev/null
    scp -o StrictHostKeyChecking=no /tmp/fxcore_code.tgz "root@$IP:/tmp/" 2>/dev/null
    scp -o StrictHostKeyChecking=no /tmp/fxcore_parquets.tgz "root@$IP:/tmp/" 2>/dev/null
    ssh -o StrictHostKeyChecking=no "root@$IP" \
        "cd /root/fx-core && tar xzf /tmp/fxcore_code.tgz && tar xzf /tmp/fxcore_parquets.tgz && ls data/m5_ohlc/*.parquet | wc -l" 2>&1 | tail -1
    echo "  ✓ $IP provisioned"
}

export -f provision_one
pids=()
for s in "${SERVERS[@]}"; do
    IP=$(echo "$s" | cut -d: -f2)
    provision_one "$IP" &
    pids+=($!)
done
# Wait for all
for p in "${pids[@]}"; do
    wait "$p" || echo "  provision pid $p failed"
done

hcloud volume detach "$VOLUME" 2>&1 | tail -1
MSG "✅ All $N servers provisioned. Sharding $((${#CAND_ARR[@]} * 48)) jobs..."

# 6. Build + shard job list
> /tmp/slot4_jobs.txt
for cand in "${CAND_ARR[@]}"; do
    for pair in "${PAIRS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "$cand $pair $seed" >> /tmp/slot4_jobs.txt
        done
    done
done
TOTAL=$(wc -l < /tmp/slot4_jobs.txt)
echo "Total jobs: $TOTAL"

for i in $(seq 0 $((N-1))); do
    awk -v n=$N -v idx=$i 'NR % n == idx' /tmp/slot4_jobs.txt > /tmp/slot4_jobs_$i.txt
    IP=$(echo "${SERVERS[$i]}" | cut -d: -f2)
    scp -o StrictHostKeyChecking=no /tmp/slot4_jobs_$i.txt "root@$IP:/tmp/jobs.txt" 2>/dev/null

    ssh -o StrictHostKeyChecking=no "root@$IP" "cat > /root/run_jobs.sh" <<'EOF_RUNNER'
#!/bin/bash
cd /root/fx-core
mkdir -p research/experiments/cma_5in/indicator_loop/results
# 8 concurrent CMAs × 2 inner workers each = 16 CPU saturated on cx53
CONCURRENCY=8
pids=()
while IFS=" " read -r cand pair seed; do
    while [ ${#pids[@]} -ge $CONCURRENCY ]; do
        new_pids=()
        for p in "${pids[@]}"; do
            kill -0 $p 2>/dev/null && new_pids+=($p)
        done
        pids=("${new_pids[@]}")
        [ ${#pids[@]} -ge $CONCURRENCY ] && sleep 5
    done
    log="research/experiments/cma_5in/indicator_loop/results/slot4_${cand}_${pair}_s${seed}.log"
    nohup python3 research/experiments/cma_5in/indicator_loop/test_slot4_swap_cma.py \
        --candidate "$cand" --pair "$pair" --seed "$seed" \
        --gens 200 --pop 40 --workers 2 > "$log" 2>&1 &
    pids+=($!)
done < /tmp/jobs.txt
wait
echo "All jobs done on $(hostname) at $(date)"
EOF_RUNNER
    ssh -o StrictHostKeyChecking=no "root@$IP" "chmod +x /root/run_jobs.sh && nohup /root/run_jobs.sh > /root/run_jobs.log 2>&1 &"
    echo "  server $i launched ($(wc -l < /tmp/slot4_jobs_$i.txt) jobs)"
done

# 7. Save server list
printf "%s\n" "${SERVERS[@]}" > research/experiments/cma_5in/indicator_loop/tier2_servers.txt

echo ""
echo "=== Deploy complete. $N servers running $TOTAL jobs. ==="
WALL=$(echo "$TOTAL / ($N * 4) * 3.5" | bc)
MSG "⚙ Tier-2 launched: $TOTAL jobs × $N servers × 4 concurrent = ~${WALL} min wall. Collect: collect_slot4_full.sh --keep"
