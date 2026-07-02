#!/bin/bash
# Resume tier-2 deploy: create missing servers 4-10, provision all 10, launch jobs.
# Assumes slot4-tier2-1, -2, -3 already created and running.
set -uo pipefail

cd "$(dirname "$0")/../../../.."

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
print(' '.join([c['name'] for c in d['candidates'] if c.get('port_status') == 'ported']))
")
CAND_ARR=($CANDS)

MSG() {
    python3 -c "import sys; sys.path.insert(0,'.'); from lib.notify import _send; _send('''$1''')"
}

# 1. Create missing servers with retry
echo "=== Creating missing servers 4-10 (retry up to 3× each) ==="
for i in 4 5 6 7 8 9 10; do
    NAME="slot4-tier2-$i"
    if hcloud server describe "$NAME" >/dev/null 2>&1; then
        echo "  $NAME already exists, skipping create"
        continue
    fi
    for attempt in 1 2 3; do
        if hcloud server create --name "$NAME" --type "$TYPE" --image ubuntu-24.04 \
             --location "$LOCATION" --ssh-key "$SSH_KEY" >/dev/null 2>&1; then
            IP=$(hcloud server ip "$NAME")
            echo "  ✓ [$i/10] $NAME $IP (attempt $attempt)"
            break
        else
            echo "  ✗ [$i/10] $NAME create attempt $attempt failed, retrying..."
            sleep 5
        fi
    done
done

# 2. Gather all server IPs
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
echo "=== $N servers online ==="
for s in "${SERVERS[@]}"; do echo "  $s"; done

if [ "$N" -lt "10" ]; then
    MSG "⚠ Tier-2: only $N/10 servers running. Proceeding with what we have."
fi

# 3. Provision server 1 with data
FIRST_NAME=$(echo "${SERVERS[0]}" | cut -d: -f1)
FIRST_IP=$(echo "${SERVERS[0]}" | cut -d: -f2)
echo ""
echo "=== Provisioning server 1 ($FIRST_IP) with data from volume ==="
hcloud volume attach "$VOLUME" --server "$FIRST_NAME" 2>&1 | tail -2
sleep 15

ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 "root@$FIRST_IP" bash <<EOF
set -e
apt-get -qq update
apt-get -qq install -y git python3-pip rsync >/dev/null 2>&1
mkdir -p /mnt/neat-data
mount /dev/disk/by-id/scsi-0HC_Volume_105213043 /mnt/neat-data 2>/dev/null || true
[ -d /root/fx-core ] || git clone $REPO /root/fx-core
cd /root/fx-core && pip install -q cma numpy pandas numba pyarrow python-dotenv requests cmaes
mkdir -p /root/fx-core/data/m5_ohlc
cp /mnt/neat-data/*_M5_kalman10_causal.parquet /root/fx-core/data/m5_ohlc/
count=\$(ls /root/fx-core/data/m5_ohlc/*_M5_kalman10_causal.parquet 2>/dev/null | wc -l)
echo "Causal parquets on disk: \$count"
[ "\$count" = "12" ] || { echo "ERROR: expected 12, got \$count"; exit 1; }
EOF
MSG "📦 Server 1 provisioned + 12 causal parquets pulled from volume"

# 4. Distribute to other servers in parallel
echo ""
echo "=== Distributing to ${#SERVERS[@]}-1 other servers in parallel ==="
for s in "${SERVERS[@]:1}"; do
    IP=$(echo "$s" | cut -d: -f2)
    (
        ssh -o StrictHostKeyChecking=no "root@$IP" "
            apt-get -qq update >/dev/null 2>&1
            apt-get -qq install -y git python3-pip rsync >/dev/null 2>&1
            pip install -q cma numpy pandas numba pyarrow python-dotenv requests cmaes
            mkdir -p /root/fx-core
        " 2>/dev/null
        ssh -o StrictHostKeyChecking=no "root@$FIRST_IP" "rsync -az --exclude='.git' -e 'ssh -o StrictHostKeyChecking=no' /root/fx-core/ root@$IP:/root/fx-core/"
        echo "  ✓ $IP ready"
    ) &
done
wait
hcloud volume detach "$VOLUME" 2>&1 | tail -1
MSG "✅ All $N servers ready. Sharding + launching $((${#CAND_ARR[@]} * 48)) jobs..."

# 5. Build + shard job list
> /tmp/slot4_jobs.txt
for cand in "${CAND_ARR[@]}"; do
    for pair in "${PAIRS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "$cand $pair $seed" >> /tmp/slot4_jobs.txt
        done
    done
done
echo "Total jobs: $(wc -l < /tmp/slot4_jobs.txt)"

# 6. Shard + distribute
for i in $(seq 0 $((N-1))); do
    awk -v n=$N -v idx=$i 'NR % n == idx' /tmp/slot4_jobs.txt > /tmp/slot4_jobs_$i.txt
    IP=$(echo "${SERVERS[$i]}" | cut -d: -f2)
    scp -o StrictHostKeyChecking=no /tmp/slot4_jobs_$i.txt "root@$IP:/tmp/jobs.txt" 2>/dev/null
    # Write runner
    ssh -o StrictHostKeyChecking=no "root@$IP" "cat > /root/run_jobs.sh" <<'EOF_RUNNER'
#!/bin/bash
cd /root/fx-core
mkdir -p research/experiments/cma_5in/indicator_loop/results
CONCURRENCY=4
pids=()
while IFS=" " read -r cand pair seed; do
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
    ssh -o StrictHostKeyChecking=no "root@$IP" "chmod +x /root/run_jobs.sh && nohup /root/run_jobs.sh > /root/run_jobs.log 2>&1 &"
    echo "  Server $i launched ($(wc -l < /tmp/slot4_jobs_$i.txt) jobs)"
done

# 7. Save server list
printf "%s\n" "${SERVERS[@]}" > research/experiments/cma_5in/indicator_loop/tier2_servers.txt

echo ""
echo "=== Deploy complete. $N servers running with sharded jobs. ==="
MSG "⚙ Tier-2 jobs launched on $N servers (~$(echo "${#CAND_ARR[@]} * 48 / ($N * 4) * 3.5" | bc) min wall). Monitor: bash collect_slot4_full.sh --keep"
