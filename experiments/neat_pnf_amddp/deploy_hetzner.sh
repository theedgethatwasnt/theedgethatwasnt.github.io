#!/usr/bin/env bash
# =============================================================================
# deploy_hetzner.sh — provision Hetzner + launch the NEAT P&F + AMDDP5 campaign.
# =============================================================================
# Follows the project Hetzner SOP (CLAUDE.md § "Hetzner Cloud Training SOP"):
#   hcloud, server type cx53, --location hel1, ssh-key "user@host".
#
# WHAT IT DOES
#   1. Creates ~4-5 cx53 servers (ubuntu-24.04, hel1).
#   2. Ships ONLY what's needed (NOT raw S5 — the prebuilt ~6.5MB box parquet instead):
#        experiment dir (phase1_harness.py, campaign.py, collect_winners.py,
#        monitor.py, the two .ini configs, cache/GBP_JPY_pnf_box_rev3.parquet)
#        + lib/fast_eval.py + lib/pnf_engine.py + research/experiments/amddp5/scorer.py.
#   3. Installs deps (python3-venv; neat-python numba numpy pandas pyarrow matplotlib requests).
#   4. Launches the 16 islands = 4 seeds x 4 exponents {0.4,0.5,0.6,0.7}, distributed
#        ~4 islands/server, PLUS an equal set of --surrogate (null) runs, nohup'd with logs.
#   5. Starts monitor.py on server 1 (headless Agg → Telegram).
#   6. Prints how to collect winners + DELETE servers (SOP: ALWAYS delete when done).
#
# ⚠️ THIS SCRIPT DOES NOT RUN ANY LIVE hcloud/ssh CALLS BY DEFAULT.
#    It is a DRY-RUN scaffold: every provisioning command is wrapped in `run()`,
#    which only ECHOES unless you export GO=1. Review, then `GO=1 ./deploy_hetzner.sh`.
#
# PREREQUISITES (export before running with GO=1):
#    OANDA_API_KEY, OANDA_ACCOUNT_ID_* ... (only if a step fetches data; here we ship
#       the prebuilt parquet, so OANDA creds are NOT required for training).
#    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID  (so monitor.py can push graphs).
#    hcloud CLI configured (the SOP API key/context is already set on `trader`).
#    SSH key "user@host" registered with Hetzner (per SOP).
# =============================================================================
set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
SERVER_TYPE="cx53"
IMAGE="ubuntu-24.04"
LOCATION="hel1"
SSH_KEY="user@host"
SERVER_PREFIX="neat-pnf"
NUM_SERVERS="${NUM_SERVERS:-4}"          # 4-5 per SOP; 16 islands / 4 servers = 4 islands/server
PAIR="GBP_JPY"
CONFIG="neat_pnf_generous.ini"
GENS="${GENS:-400}"
SEEDS=(0 1 2 3)                          # 4 seeds
EXPONENTS=(0.4 0.5 0.6 0.7)              # 4 exponents → 4x4 = 16 islands
REMOTE_DIR="/root/neat_pnf"              # working dir on each server
REMOTE_VENV="/root/venv"

# Local paths (relative to this script's repo) — only these get shipped.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
EXP_FILES=(
  "$HERE/phase1_harness.py"
  "$HERE/campaign.py"
  "$HERE/collect_winners.py"
  "$HERE/monitor.py"
  "$HERE/neat_pnf_generous.ini"
  "$HERE/neat_pnf_2in_3out.ini"
)
CACHE_FILE="$HERE/cache/${PAIR}_pnf_box_rev3.parquet"   # ~6.5MB prebuilt box series
LIB_FILES=(
  "$REPO_ROOT/lib/fast_eval.py"
  "$REPO_ROOT/lib/pnf_engine.py"
)
SCORER_FILE="$REPO_ROOT/research/experiments/amddp5/scorer.py"

# ── run(): dry-run wrapper. Echoes unless GO=1. ────────────────────────────────
GO="${GO:-0}"
run() {
  if [[ "$GO" == "1" ]]; then
    echo "+ $*"
    "$@"
  else
    echo "DRY-RUN> $*"
  fi
}
# ssh_run(): same wrapper for remote commands.
ssh_run() {  # ssh_run <ip> <command-string>
  local ip="$1"; shift
  if [[ "$GO" == "1" ]]; then
    echo "+ ssh root@$ip $*"
    ssh -o StrictHostKeyChecking=accept-new "root@$ip" "$*"
  else
    echo "DRY-RUN> ssh root@$ip '$*'"
  fi
}

# ── 0. Sanity: local files exist ───────────────────────────────────────────────
echo "=== 0. Verifying local payload exists ==="
missing=0
for f in "${EXP_FILES[@]}" "$CACHE_FILE" "${LIB_FILES[@]}" "$SCORER_FILE"; do
  if [[ -f "$f" ]]; then
    printf '  ok  %s (%s)\n' "$f" "$(du -h "$f" | cut -f1)"
  else
    printf '  MISSING  %s\n' "$f"
    missing=1
  fi
done
if [[ "$missing" == "1" ]]; then
  echo "ERROR: payload incomplete — fix paths before deploying." >&2
  exit 1
fi
echo "  (NOTE: raw S5 is intentionally NOT shipped — only the prebuilt box parquet.)"
echo

# ── 1. Create servers (idempotent-ish: skip if already exists) ─────────────────
echo "=== 1. Creating $NUM_SERVERS x $SERVER_TYPE ($IMAGE, $LOCATION) ==="
for i in $(seq 1 "$NUM_SERVERS"); do
  name="${SERVER_PREFIX}-${i}"
  if hcloud server describe "$name" >/dev/null 2>&1; then
    echo "  $name already exists — skip create"
  else
    run hcloud server create --name "$name" --type "$SERVER_TYPE" \
        --image "$IMAGE" --location "$LOCATION" --ssh-key "$SSH_KEY"
  fi
done
echo

# ── Resolve server IPs (only meaningful with GO=1) ─────────────────────────────
declare -a IPS=()
echo "=== Resolving server IPs ==="
for i in $(seq 1 "$NUM_SERVERS"); do
  name="${SERVER_PREFIX}-${i}"
  if [[ "$GO" == "1" ]]; then
    ip="$(hcloud server ip "$name")"
  else
    ip="<ip-of-${name}>"
  fi
  IPS+=("$ip")
  echo "  $name -> $ip"
done
echo

# ── 2. Provision each server: deps + payload ───────────────────────────────────
echo "=== 2. Provisioning servers (deps + payload via rsync) ==="
PROVISION_CMD=$(cat <<'EOSH'
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-venv python3-pip rsync >/dev/null
python3 -m venv /root/venv
/root/venv/bin/pip install --quiet --upgrade pip
/root/venv/bin/pip install --quiet neat-python numba numpy pandas pyarrow matplotlib requests
mkdir -p /root/neat_pnf/research/experiments/neat_pnf_amddp/cache
mkdir -p /root/neat_pnf/research/experiments/amddp5
mkdir -p /root/neat_pnf/lib
# package __init__ shims so `from research.experiments...` and `from lib...` import cleanly
touch /root/neat_pnf/research/__init__.py
touch /root/neat_pnf/research/experiments/__init__.py
touch /root/neat_pnf/research/experiments/neat_pnf_amddp/__init__.py
touch /root/neat_pnf/research/experiments/amddp5/__init__.py
touch /root/neat_pnf/lib/__init__.py
EOSH
)

for idx in "${!IPS[@]}"; do
  ip="${IPS[$idx]}"
  n=$((idx + 1))
  echo "--- server ${SERVER_PREFIX}-${n} ($ip) ---"
  ssh_run "$ip" "$PROVISION_CMD"

  # rsync the experiment files into the mirrored repo layout on the server
  EXP_REMOTE="$REMOTE_DIR/research/experiments/neat_pnf_amddp"
  for f in "${EXP_FILES[@]}"; do
    run rsync -az -e "ssh -o StrictHostKeyChecking=accept-new" \
        "$f" "root@${ip}:${EXP_REMOTE}/"
  done
  run rsync -az -e "ssh -o StrictHostKeyChecking=accept-new" \
      "$CACHE_FILE" "root@${ip}:${EXP_REMOTE}/cache/"
  for f in "${LIB_FILES[@]}"; do
    run rsync -az -e "ssh -o StrictHostKeyChecking=accept-new" \
        "$f" "root@${ip}:${REMOTE_DIR}/lib/"
  done
  run rsync -az -e "ssh -o StrictHostKeyChecking=accept-new" \
      "$SCORER_FILE" "root@${ip}:${REMOTE_DIR}/research/experiments/amddp5/"
done
echo

# ── 3. Launch islands — 16 real + 16 surrogate, distributed ~4 islands/server ──
# island index = seed_index*4 + exp_index  (0..15). Server = island % NUM_SERVERS.
echo "=== 3. Launching 16 real + 16 surrogate islands across $NUM_SERVERS servers ==="
ENV_EXPORTS="export TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN:-} TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID:-}"

launch_one() {  # launch_one <ip> <island> <seed> <exp> <surrogate:0|1>
  local ip="$1" island="$2" seed="$3" exp="$4" sur="$5"
  local sur_flag="" tag="isl${island}_seed${seed}_exp${exp}"
  if [[ "$sur" == "1" ]]; then sur_flag="--surrogate"; tag="${tag}_surrogate"; fi
  local log="$REMOTE_DIR/run_${tag}.log"
  # PYTHONPATH=$REMOTE_DIR so `from research...`/`from lib...` resolve; PYTHONUNBUFFERED for live logs.
  local cmd="cd $REMOTE_DIR && $ENV_EXPORTS && \
PYTHONPATH=$REMOTE_DIR PYTHONUNBUFFERED=1 \
nohup $REMOTE_VENV/bin/python3 research/experiments/neat_pnf_amddp/campaign.py \
  --island $island --seed $seed --exp $exp \
  --config $CONFIG --gens $GENS --pair $PAIR $sur_flag \
  > $log 2>&1 &"
  ssh_run "$ip" "$cmd"
}

island=0
for si in "${!SEEDS[@]}"; do
  for ei in "${!EXPONENTS[@]}"; do
    seed="${SEEDS[$si]}"
    exp="${EXPONENTS[$ei]}"
    srv=$(( island % NUM_SERVERS ))
    ip="${IPS[$srv]}"
    echo "  island $island (seed=$seed exp=$exp) REAL      -> ${SERVER_PREFIX}-$((srv+1)) ($ip)"
    launch_one "$ip" "$island" "$seed" "$exp" 0
    echo "  island $island (seed=$seed exp=$exp) SURROGATE -> ${SERVER_PREFIX}-$((srv+1)) ($ip)"
    launch_one "$ip" "$island" "$seed" "$exp" 1
    island=$((island + 1))
  done
done
echo "  (32 processes = 16 real + 16 surrogate; ~8 procs/server on 4 servers.)"
echo

# ── 4. Start the headless monitor on server 1 ──────────────────────────────────
echo "=== 4. Starting monitor.py on ${SERVER_PREFIX}-1 (headless Agg → Telegram) ==="
MON_IP="${IPS[0]}"
MON_CMD="cd $REMOTE_DIR && $ENV_EXPORTS && \
PYTHONPATH=$REMOTE_DIR PYTHONUNBUFFERED=1 \
nohup $REMOTE_VENV/bin/python3 research/experiments/neat_pnf_amddp/monitor.py \
  --runs research/experiments/neat_pnf_amddp/campaign_runs --pair $PAIR \
  --interval-min 10 --interval-gens 10 \
  > $REMOTE_DIR/monitor.log 2>&1 &"
ssh_run "$MON_IP" "$MON_CMD"
echo "  NOTE: monitor on server 1 only sees server-1 islands' campaign_runs/."
echo "        To aggregate ALL servers, periodically rsync each server's campaign_runs/"
echo "        to one host and run monitor.py there, e.g.:"
echo "          for ip in ${IPS[*]}; do"
echo "            rsync -az root@\$ip:$REMOTE_DIR/research/experiments/neat_pnf_amddp/campaign_runs/ ./campaign_runs/"
echo "          done && python3 monitor.py --runs ./campaign_runs --once"
echo

# ── 5. Final instructions: collect winners + DELETE servers ────────────────────
cat <<EOF
=============================================================================
LAUNCHED.  ~3-4h for 400 gens (early-stop ~60 gens stagnation may finish sooner).

MONITOR
  ssh root@${IPS[0]} 'tail -f $REMOTE_DIR/monitor.log'
  ssh root@${IPS[0]} 'tail -f $REMOTE_DIR/run_isl0_seed0_exp0.4.log'
  Telegram pushes 6 graphs every ~10 gens / 10 min.

WHEN DONE — collect all islands' best bundles to ONE host, then select the winner:
  mkdir -p ./campaign_runs
  for ip in ${IPS[*]}; do
    rsync -az root@\$ip:$REMOTE_DIR/research/experiments/neat_pnf_amddp/campaign_runs/ ./campaign_runs/
  done
  python3 research/experiments/neat_pnf_amddp/collect_winners.py \\
      --root ./campaign_runs/$PAIR --emit ./selected_winner.pkl
  # collect_winners ranks REAL islands by val amddp/day and compares to the
  # best SURROGATE (the equal-compute null). Then run the ONE sealed OOS eval
  # on selected_winner.pkl (R8: touch test exactly once).

⚠️ ALWAYS DELETE SERVERS WHEN FINISHED (SOP — they bill hourly):
  for i in \$(seq 1 $NUM_SERVERS); do hcloud server delete "${SERVER_PREFIX}-\$i"; done
  # (neat-data volume is NOT used here — nothing persistent to detach.)
=============================================================================
EOF

if [[ "$GO" != "1" ]]; then
  echo
  echo ">>> DRY-RUN complete. No servers created, no commands executed."
  echo ">>> Review the plan above, then run:   GO=1 $0"
fi
