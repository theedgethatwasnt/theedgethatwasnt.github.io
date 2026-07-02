# Experiment: Per-Pair Custom Training

## Hypothesis
The current ASI-MC genome was trained on EUR_JPY only and generalized to all 12 pairs.
Per-pair training would optimize each genome for its pair's specific characteristics
(volatility, spread, session patterns, ASI behavior). Could yield significantly better
per-pair performance at the cost of managing 12 genomes instead of 1.

## Design

### Training (per pair)
For each of the 12 pairs:
1. Load that pair's exported indicator parquet
2. Pretrain on sine wave (50 gens)
3. Continue on that pair's data (150 gens)
4. Island NEAT: 4 islands × 150 pop
5. Save best genome as `asi_mc_{pair}_best.pkl`

### Deployment options
**Option A: 12 containers** — one per pair per account
- Most isolated, but 12 × containers is heavy
- Each container subscribes to one pair only

**Option B: 1 container, genome-per-pair routing**
- Single container loads 12 genomes
- Routes: `if pair == "EUR_JPY": use genome_eurjpy`
- Lighter footprint, more complex code

**Option C: Multi-genome env var**
- Already supported: `NEAT_GENOME=eur_jpy.pkl,usd_jpy.pkl,...`
- Container maps pair → genome internally

Recommend **Option C** — already built into strategy_neat.

### Comparison
- **Baseline**: Single genome (current ASI-MC v2) on all 12 pairs
- **Test**: 12 custom genomes, each on its trained pair
- Metrics: per-pair pips/day, overall pips/day, WR, Sharpe

### Hetzner plan
- 12 pairs ÷ 5 servers = ~2-3 pairs per server
- Each pair: ~10 min training (just 1 pair, small data)
- Total: ~30 min on 5 servers, ~$0.50

### Validation
- Per-pair WF (3 splits) — each genome only needs to pass on its own pair
- MC shuffle per pair
- Compare aggregate vs single-genome aggregate

## Risk
- 12 genomes = 12× maintenance burden
- Some pairs may not have enough signal for a standalone genome
- Overfitting risk: genome memorizes one pair's patterns

## Estimated effort
- Training script modification: 30 min
- Hetzner run: 30 min
- Validation: 30 min
- Total: ~1.5 hours
