# Fixed vs Free Topology — Should We Let NEAT Evolve Morphology?

**Date**: 2026-04-05

---

## Current State: Fixed Topology (IronNet)

We deliberately fixed the topology (4→4→3 for IronNet, 4→5→7→3 for activation experiments) and only mutate weights, biases, and activations.

**Why we fixed it:**
- Prevents premature topology bloat (NEAT's node-add mutation can create very deep useless chains)
- More reproducible — same architecture each run
- Faster convergence — fewer degrees of freedom
- Easier to analyse (we know exactly what layer does what)
- Innovation number collisions were causing pickle failures in distributed training

**The cost:**
- We may be constraining the network to a suboptimal depth/width
- A wider L1 or deeper network might find better solutions
- Fixed topology means every node must justify its existence (no redundancy)

---

## The Case for Free Topology

Standard NEAT (Neuroevolution of Augmenting Topologies) starts minimal and grows:
- Gen 0: direct input→output connections only (linear model)
- Mutations add nodes and connections over time
- Speciation protects innovation — new topologies compete within species before competing globally
- Resulting networks: often surprisingly minimal (1-3 hidden nodes for simple tasks)

Benefits:
- Finds **minimum sufficient complexity** — only adds nodes if they help
- Can discover non-obvious architectures (e.g., skip connections naturally emerge)
- Less hyperparameter sensitivity (no need to choose hidden layer size)

Risks:
- Much larger search space — convergence takes 2-5× more generations
- Innovation number management is tricky in distributed/island setting
- Hard to guarantee fixed-depth inference in live trading container

---

## Recommendation: Run Both in Parallel on Hetzner

| Experiment | Topology | Gens | Servers | Purpose |
|-----------|---------|------|---------|---------|
| Fixed v5  | 5→5→7→3 fixed | 200 | 2 | Baseline with AE inputs + 13 activations |
| Free NEAT | minimal→grows | 300 | 2 | Find optimal depth/width from scratch |

**Compare:**
- Free NEAT final topology depth vs fixed: is 2 layers actually optimal?
- Free NEAT likely converges to 1-3 hidden nodes — if so, our fixed 12 nodes is overkill
- If free NEAT finds >2 layers naturally, that validates deeper fixed topology

**Practical constraint for live deployment:**
Whatever topology wins, it must run in `neat.nn.FeedForwardNetwork.create()` in the strategy container. Both fixed and free NEAT produce this — no deployment difference.

---

## What Free NEAT Needs

1. Standard NEAT config (not IronNet's fixed config):
   - `conn_add_prob = 0.3`
   - `node_add_prob = 0.1`
   - `conn_delete_prob = 0.1`
   - `node_delete_prob = 0.05`
   - Start with `initial_connection = partial_direct 0.5`

2. Speciation enabled (already default in neat-python)

3. More generations (300 vs 200) — topology search takes longer

4. Innovation number collision fix for distributed training (the Lock bug from V3 training — needs proper `_GlobalNodeIDFactory` using a file lock or pre-allocated ranges per island)

---

## Expected Outcome

Based on the activation study findings:
- Our 4-input problems are not very complex (sine wave reconstruction, momentum trading)
- Free NEAT will likely find 1-2 hidden nodes sufficient
- If the AE latent space is truly orthogonal and informative, even simpler networks may work
- The latent representation does the heavy lifting; NEAT just needs to learn thresholds

This would suggest: **AE + simple network (1-3 nodes) beats complex fixed topology** — the compression is the key innovation, not the network depth.
