#!/bin/bash
# Launch Path B + Path C sweeps in parallel on 4 existing servers.
# Assumes: servers <SERVER_IP_15>, <SERVER_IP_16> (hel1), <SERVER_IP_17>, <SERVER_IP_18> (nbg1)
# already provisioned with code + parquets (from tier-2 setup).
set -uo pipefail

cd "$(dirname "$0")/../../../.."

# 2 servers per path
PATH_B_IPS=("<SERVER_IP_15>" "<SERVER_IP_16>")
PATH_C_IPS=("<SERVER_IP_17>" "<SERVER_IP_18>")

# Path B excludes the core indicators from swap list
PATH_B_CORE="cci bb_width ema21_ratio atr_ratio"
CANDS=$(python3 -c "
import json
d = json.load(open('research/experiments/cma_5in/indicator_loop/candidates.json'))
print(' '.join([c['name'] for c in d['candidates'] if c.get('port_status') == 'ported']))
")
CAND_ARR=($CANDS)
# Path B candidates: all ported minus the 4 core
PATH_B_CANDS=()
for c in "${CAND_ARR[@]}"; do
    if [[ ! " $PATH_B_CORE " =~ " $c " ]]; then
        PATH_B_CANDS+=("$c")
    fi
done
# Path C: all ported (V3 core is different)
PATH_C_CANDS=("${CAND_ARR[@]}")

PAIRS=(EUR_JPY USD_JPY)
SEEDS=(42 137)

MSG() {
    python3 -c "import sys; sys.path.insert(0,'.'); from lib.notify import _send; _send('''$1''')"
}

echo "Path B: ${#PATH_B_CANDS[@]} candidates × 2 pairs × 2 seeds = $((${#PATH_B_CANDS[@]} * 4)) jobs on ${#PATH_B_IPS[@]} servers"
echo "Path C: ${#PATH_C_CANDS[@]} candidates × 2 pairs × 2 seeds = $((${#PATH_C_CANDS[@]} * 4)) jobs on ${#PATH_C_IPS[@]} servers"

# Build + shard job lists
> /tmp/pathB_jobs.txt
for cand in "${PATH_B_CANDS[@]}"; do
    for pair in "${PAIRS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "B $cand $pair $seed" >> /tmp/pathB_jobs.txt
        done
    done
done
> /tmp/pathC_jobs.txt
for cand in "${PATH_C_CANDS[@]}"; do
    for pair in "${PAIRS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "C $cand $pair $seed" >> /tmp/pathC_jobs.txt
        done
    done
done

# Copy new runner script + candidates.json to all servers
for IP in "${PATH_B_IPS[@]}" "${PATH_C_IPS[@]}"; do
    scp -o StrictHostKeyChecking=no research/experiments/cma_5in/indicator_loop/test_paths_bc_cma.py \
        "root@$IP:/root/fx-core/research/experiments/cma_5in/indicator_loop/" 2>&1 | tail -1 &
done
wait

# Shard + launch Path B on its 2 servers
NB=${#PATH_B_IPS[@]}
for i in $(seq 0 $((NB-1))); do
    awk -v n=$NB -v idx=$i 'NR % n == idx' /tmp/pathB_jobs.txt > /tmp/pathB_$i.txt
    IP="${PATH_B_IPS[$i]}"
    scp -o StrictHostKeyChecking=no /tmp/pathB_$i.txt "root@$IP:/tmp/jobs_B.txt"
    ssh -o StrictHostKeyChecking=no "root@$IP" "cat > /root/run_pathB.sh" <<'EOF_B'
#!/bin/bash
cd /root/fx-core
mkdir -p research/experiments/cma_5in/indicator_loop/results
CONCURRENCY=8
pids=()
while IFS=" " read -r path cand pair seed; do
    while [ ${#pids[@]} -ge $CONCURRENCY ]; do
        new_pids=()
        for p in "${pids[@]}"; do
            kill -0 $p 2>/dev/null && new_pids+=($p)
        done
        pids=("${new_pids[@]}")
        [ ${#pids[@]} -ge $CONCURRENCY ] && sleep 5
    done
    log="research/experiments/cma_5in/indicator_loop/results/path${path}_${cand}_${pair}_s${seed}.log"
    nohup python3 research/experiments/cma_5in/indicator_loop/test_paths_bc_cma.py \
        --path "$path" --candidate "$cand" --pair "$pair" --seed "$seed" \
        --gens 100 --pop 40 --workers 2 > "$log" 2>&1 &
    pids+=($!)
done < /tmp/jobs_B.txt
wait
echo "Path B done on $(hostname) at $(date)"
EOF_B
    ssh -o StrictHostKeyChecking=no "root@$IP" "chmod +x /root/run_pathB.sh && nohup /root/run_pathB.sh > /root/run_pathB.log 2>&1 &"
    echo "  Path B server $i launched ($(wc -l < /tmp/pathB_$i.txt) jobs)"
done

# Shard + launch Path C on its 2 servers
NC=${#PATH_C_IPS[@]}
for i in $(seq 0 $((NC-1))); do
    awk -v n=$NC -v idx=$i 'NR % n == idx' /tmp/pathC_jobs.txt > /tmp/pathC_$i.txt
    IP="${PATH_C_IPS[$i]}"
    scp -o StrictHostKeyChecking=no /tmp/pathC_$i.txt "root@$IP:/tmp/jobs_C.txt"
    ssh -o StrictHostKeyChecking=no "root@$IP" "cat > /root/run_pathC.sh" <<'EOF_C'
#!/bin/bash
cd /root/fx-core
mkdir -p research/experiments/cma_5in/indicator_loop/results
CONCURRENCY=8
pids=()
while IFS=" " read -r path cand pair seed; do
    while [ ${#pids[@]} -ge $CONCURRENCY ]; do
        new_pids=()
        for p in "${pids[@]}"; do
            kill -0 $p 2>/dev/null && new_pids+=($p)
        done
        pids=("${new_pids[@]}")
        [ ${#pids[@]} -ge $CONCURRENCY ] && sleep 5
    done
    log="research/experiments/cma_5in/indicator_loop/results/path${path}_${cand}_${pair}_s${seed}.log"
    nohup python3 research/experiments/cma_5in/indicator_loop/test_paths_bc_cma.py \
        --path "$path" --candidate "$cand" --pair "$pair" --seed "$seed" \
        --gens 100 --pop 40 --workers 2 > "$log" 2>&1 &
    pids+=($!)
done < /tmp/jobs_C.txt
wait
echo "Path C done on $(hostname) at $(date)"
EOF_C
    ssh -o StrictHostKeyChecking=no "root@$IP" "chmod +x /root/run_pathC.sh && nohup /root/run_pathC.sh > /root/run_pathC.log 2>&1 &"
    echo "  Path C server $i launched ($(wc -l < /tmp/pathC_$i.txt) jobs)"
done

MSG "⚙ Paths B + C launched. 4 servers × 8 concurrent. ETA ~2h."
echo "Deploy complete."
