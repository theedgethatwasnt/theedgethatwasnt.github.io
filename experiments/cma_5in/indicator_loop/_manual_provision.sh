#!/bin/bash
# Manual provision + launch for existing servers whose parquets are in /tmp
# Usage: _manual_provision.sh IP1 IP2 ...
set -uo pipefail

cd "$(dirname "$0")/../../../.."
CANDS=$(python3 -c "
import json
d = json.load(open('research/experiments/cma_5in/indicator_loop/candidates.json'))
print(' '.join([c['name'] for c in d['candidates'] if c.get('port_status') == 'ported']))
")
CAND_ARR=($CANDS)
PAIRS=(EUR_USD GBP_USD USD_JPY AUD_USD EUR_JPY GBP_JPY AUD_JPY CAD_JPY CHF_JPY NZD_JPY NZD_USD EUR_GBP)
SEEDS=(42 137 23 7)

IPS=("$@")
N=${#IPS[@]}
echo "Manual provision for $N servers with ${#CAND_ARR[@]} candidates, jobs=$((${#CAND_ARR[@]} * 48))"

# Build job queue
> /tmp/slot4_jobs.txt
for cand in "${CAND_ARR[@]}"; do
    for pair in "${PAIRS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "$cand $pair $seed" >> /tmp/slot4_jobs.txt
        done
    done
done

# Provision each in parallel
pids=()
for i in $(seq 0 $((N-1))); do
    IP="${IPS[$i]}"
    awk -v n=$N -v idx=$i 'NR % n == idx' /tmp/slot4_jobs.txt > /tmp/slot4_jobs_$i.txt
    (
        echo "[$IP] provisioning..."
        ssh -o StrictHostKeyChecking=no "root@$IP" "apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y python3-pip rsync >/dev/null 2>&1"
        ssh -o StrictHostKeyChecking=no "root@$IP" "pip install --break-system-packages -q cma numpy pandas numba pyarrow python-dotenv requests cmaes 2>&1 | tail -2"
        # Ensure tgz present (scp if missing)
        if ! ssh -o StrictHostKeyChecking=no "root@$IP" "test -f /tmp/fxcore_code.tgz && test -f /tmp/fxcore_parquets.tgz" 2>/dev/null; then
            echo "[$IP] scp'ing tgzs..."
            scp -o StrictHostKeyChecking=no /tmp/fxcore_code.tgz "root@$IP:/tmp/"
            scp -o StrictHostKeyChecking=no /tmp/fxcore_parquets.tgz "root@$IP:/tmp/"
        fi
        # Extract
        ssh -o StrictHostKeyChecking=no "root@$IP" "mkdir -p /root/fx-core && cd /root/fx-core && tar xzf /tmp/fxcore_code.tgz && tar xzf /tmp/fxcore_parquets.tgz && ls data/m5_ohlc/*_causal.parquet | wc -l"
        # Ship job list
        scp -o StrictHostKeyChecking=no /tmp/slot4_jobs_$i.txt "root@$IP:/tmp/jobs.txt"
        # Ship runner
        ssh -o StrictHostKeyChecking=no "root@$IP" "cat > /root/run_jobs.sh" <<'RUNNER_EOF'
#!/bin/bash
cd /root/fx-core
mkdir -p research/experiments/cma_5in/indicator_loop/results
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
RUNNER_EOF
        ssh -o StrictHostKeyChecking=no "root@$IP" "chmod +x /root/run_jobs.sh && nohup /root/run_jobs.sh > /root/run_jobs.log 2>&1 &"
        sleep 3
        running=$(ssh -o StrictHostKeyChecking=no "root@$IP" "ps -ef | grep test_slot4_swap | grep -v grep | wc -l")
        jobs_count=$(wc -l < /tmp/slot4_jobs_$i.txt)
        echo "[$IP] launched $jobs_count jobs, CMAs currently running: $running"
    ) &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "Provision complete."
