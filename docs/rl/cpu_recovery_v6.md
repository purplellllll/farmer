# CPU PPO recovery v6: cash-flow shaping and actor–critic isolation

## Why v5 was discarded

CPU v5 completed iterations 115–164 without a numerical failure, but none of
its three shaping stages won a scripted game.  Its final ten iterations had
nine scripted losses and one snapshot win, with mean score difference -299.1.
The update signal was also extremely small: entropy stayed near 0.016–0.018,
mean approximate KL stayed near zero, and no target-KL stop fired.  Local
benchmark artifacts additionally showed a failed checkpoint issuing repeated
seed purchases, with duplicate market-order rate 0.90 and about 800 early cash
spent.  The failure was therefore strategic credit assignment, not exploding
gradients or an illegal-action crash.

The three old potentials valued seeds, harvested inventory and cash almost
equally.  That makes harvest and sale neutral transitions and gives PPO little
guidance for finishing a production cycle.  One 143-step game per update also
made every CPU gradient depend on one seed and one seat.

## v6 reward

`cashflow_cycle_v4` orders observable asset stages by liquidity:

| State component | Weight |
| --- | ---: |
| cash | 1.00 |
| harvested product inventory | 0.80 |
| seed inventory | 0.70 |
| carried animal | 0.70 |
| placed animal capital | 0.65 |
| planted crop capital | 0.60 |
| animal output | 0.50 |
| field crop output | 0.35 |

This makes `harvest -> shed -> sell -> cash` progressively valuable.  The
per-step shaping reward remains
`c * (gamma * Phi(next_state) - Phi(state))`, and the terminal potential is
zero.  Scaling the potential by a constant does not introduce an action reward
cycle.  `realized_cash_v5` is a more conservative fallback that discounts all
illiquid assets further if the first profile still learns hoarding.

## Actor–critic and PPO changes

The 8-layer, 256-wide residual Transformer is retained (6,566,977 parameters).
The actor and critic still share its representation, but v6 scales only the
critic gradient reaching the shared encoder to 0.25; the critic head keeps its
full gradient.  The value-loss coefficient is reduced from 0.5 to 0.2.  This
is a small, checkpoint-compatible approximation to the objective isolation
motivation in PPG rather than a second full Transformer that would exceed the
current laptop's memory budget.

Each update now collects two complete 144-step games (286 observed
transitions), alternates learner seat, uses two PPO minibatch epochs, and sets
GAE lambda to 0.98.  A frozen behavior-cloning policy is retained with KL
coefficient 0.5.  The initial curriculum is 100% official scripted starter;
self-play snapshots remain disabled in practice until the scripted promotion
gate can pass.  v6 starts from `starter_bc_v5.pt`, not the collapsed CPU v5
optimizer or policy.

The choices follow the original methods rather than leaderboard anecdotes:

- [PPO](https://arxiv.org/abs/1707.06347) explicitly supports multiple epochs
  of minibatch optimization on each on-policy sample batch.
- [GAE](https://arxiv.org/abs/1506.02438) describes lambda as the bias–variance
  control for exponentially weighted advantages.
- [Potential-based reward shaping](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)
  gives the policy-invariant form used here and motivates avoiding arbitrary
  repeatable event bonuses.
- [PPG](https://proceedings.mlr.press/v139/cobbe21a.html) documents interference
  between policy and value objectives in shared representations.
- [RL with demonstrations](https://arxiv.org/abs/1709.10087) supports retaining
  demonstrations to reduce sample complexity and improve robustness.
- [AlphaStar](https://www.nature.com/articles/s41586-019-1724-z) motivates a
  supervised seed followed by diverse league training; v6 deliberately delays
  league diversity until it can beat the fixed curriculum opponent.

## Run and safety behavior

```powershell
./scripts/start_cpu_rl_recovery_v6.ps1 -Iterations 80
```

The supervisor runs one iteration per child process, records collapse windows,
and redesigns the reward instead of auto-pausing.  The initial 1.85 GiB worker
guard was too small for the first AdamW state allocation and stopped before a
checkpoint was written; the bound is now 2.10 GiB.  A memory-guard stop is an
independent system-safety failure, not a policy-collapse response.
