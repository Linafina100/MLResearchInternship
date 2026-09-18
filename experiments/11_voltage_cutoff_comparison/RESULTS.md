# Voltage-cutoff comparison: original vs. lower

## What this experiment is

Experiment 06 changed two things relative to the main pipeline at once: a
randomized discharge C-rate, and lower voltage cutoffs (LFP 1.8V->1.5V,
NMC 2.3V->1.8V). Its `RESULTS.md` flagged this as an unresolved caveat: the
*original* code comment for those cutoffs warned that below ~2.0V "both
chemistries look nearly identical," yet several of 06's surviving bins sit
in or near that region, and it was never checked whether 06's strong
accuracy reflects genuine separability there or an artifact of the lower
cutoff.

This experiment isolates that one variable: same fixed 0.6C
continuous-discharge protocol as experiment 03 (the main pipeline), only
the voltage cutoffs (and the matching feature-extraction `V_BIN_MIN`)
differ --

| Variant | LFP cutoff | NMC cutoff | V_BIN_MIN |
|---|---|---|---|
| `original` | 1.8V | 2.3V | 1.9V |
| `lower` | 1.5V | 1.8V | 1.5V |

-- run at two starting-SOC conditions: root's own default (Initial_SOC
drawn from 0.5-1.0, i.e. batteries start near full) and a low-SOC condition
pinned to 0.1-0.4 (matching experiment 06's lowest interval, the one whose
surviving bins reach down toward the flagged ~2.0V region).

Run via `compare_voltage_cutoffs.py`, which invokes root
`simulate_batteries.py`, `feature_engineering.py`, and `ml_pipeline.py`
directly (env-var/parameter overrides added to the first two for this
purpose -- see their own docstrings/signatures), rather than yet another
locally forked copy: neither root script currently has an active caller
among the other experiment sweeps, so this is the first sweep to exercise
the "main" scripts as such.

## Results

| SOC Condition | Cutoff Variant | RF Accuracy | XGB Accuracy | Surviving Bins | Top Features |
|---|---|---|---|---|---|
| default (0.5-1.0) | original | 78.12% | 83.33% | `3.3-3.2` | `3.3-3.2`:1.0000 |
| default (0.5-1.0) | lower | 79.17% | 81.25% | `3.3-3.2` | `3.3-3.2`:1.0000 |
| low SOC (0.1-0.4) | original | 94.74% | 94.74% | `3.2-3.1`, `2.4-2.3`, `2.3-2.2` | `2.4-2.3`:0.3839; `2.3-2.2`:0.3754; `3.2-3.1`:0.2406 |
| low SOC (0.1-0.4) | lower | 75.79% | 78.95% | `3.2-3.1` | `3.2-3.1`:1.0000 |

## Interpretation

**At the default SOC range, the cutoff choice made essentially no
difference** -- both variants collapsed to the identical single surviving
bin (`dV_dQ_V_3.3_3.2`), so the ~1-2pp accuracy gap between them is noise
from one weak feature, not a cutoff effect. The reason: at high starting
SOC, LFP's voltage barely leaves its plateau (~3.24V mean) while NMC sweeps
a much wider, steeper range down to its own cutoff. The mutual-coverage
filter (>=20% of *each* chemistry must have a sample in a bin) drops nearly
the whole overlap region because too few individual NMC runs leave a
recorded sample in each fine 0.1V bin during that fast, steep transition --
a bottleneck that sits well above either cutoff, so the extra low-voltage
headroom the "lower" variant opens up is never even reached in a way that
produces usable coverage.

**At low SOC, the cutoff choice mattered a lot -- but the result is the
opposite of what motivated this experiment, and it doesn't hold up under
inspection.** The `original` cutoffs scored 94.74% (both models), driven
almost entirely (76% of RF's importance) by two bins straddling NMC's exact
cutoff voltage, `2.4-2.3` and `2.3-2.2`. Checking the raw simulation data
explains why this is an artifact, not signal:

```
NMC (original, low SOC): last recorded sample mean = 2.300V (236/236
  batteries within 0.02V of the 2.3V cutoff), mean jump from the
  second-to-last sample = 0.836V (max 1.166V)
LFP (original, low SOC): last recorded sample mean = 1.800V (249/249
  within 0.02V of the 1.8V cutoff), mean jump = 0.373V (max 0.828V)
```

PyBaMM's `"Discharge until X V"` experiment is event-triggered: the solver
takes one large final step to land almost exactly on the cutoff voltage,
regardless of how far the previous sample was. For NMC under the
`original` cutoff at low starting SOC, that final step is huge (mean
0.84V, sometimes over 1V) and lands in the `2.3-2.2`/`2.4-2.3` bins for
*every single battery* -- 100% coverage, but of one artificially large
terminal dV/dQ value, not real mid-discharge signal. LFP happens to have
genuine continuous samples passing through that same voltage band on its
way to its own (lower, 1.8V) cutoff, so the bin passes the mutual-coverage
filter -- but what the classifier is actually learning is "NMC's value in
this bin is a large outlier from a single big terminal jump, LFP's is a
normal continuous value," which is a solver-termination artifact tied to
the specific cutoff voltage chosen, not a chemistry-derived electrochemical
signature.

The same table confirms this mechanism is universal, not specific to one
variant: *every* run, at *every* cutoff, has 100% of batteries landing
their last sample within 0.02V of that chemistry's cutoff, with jump sizes
ranging 0.37-1.66V depending on cutoff and SOC condition. Whether this
shows up as a "surviving" bin is incidental -- it depends on whether the
artifact bin happens to also get real coverage from the *other* chemistry's
normal trace, which is why it appeared prominently for `original`/low-SOC
(NMC's 2.3V artifact bin sits inside LFP's normal discharge range) but not
for `lower`/low-SOC (NMC's 1.8V artifact bin is very close to LFP's own
1.5V cutoff and jump zone, giving less room for LFP's normal, non-artifact
samples to populate it).

**Bottom line for the original question**: this comparison does not
support either cutoff choice as clearly better. The default-range result
is uninformative (identical single weak bin either way). The low-SOC
result superficially favors the original cutoffs, but that result is
substantially explained by a termination-step artifact rather than genuine
low-voltage separability -- which means it doesn't validate the original
cutoffs either, and it doesn't resolve experiment 06's caveat about the
lower cutoffs one way or the other.

## Broader implication: experiment 03's headline results are contaminated by this artifact (confirmed, not addressed in this pass)

This termination-step artifact is a property of the continuous-discharge
protocol itself (root `simulate_batteries.py`, used unchanged by
experiments 03/06/this one), not something introduced by this experiment.
Checked directly against both existing experiments' already-committed raw
data (still present locally under `data/continuous_soc_*/` and
`data/const_random_soc_*/`):

**Every continuous-discharge run in the repo has the same signature**:
100% of batteries (both chemistries, every SOC interval, both experiments)
land their last raw sample within 0.02V of that chemistry's cutoff, with a
large final jump (0.37-1.86V) from the second-to-last sample:

```
exp03 (cutoffs 1.8/2.3), NMC last-sample jump:  0.1-0.4: 0.84V | 0.3-0.6: 0.92V | 0.7-1.0: 1.00V
exp03 (cutoffs 1.8/2.3), LFP last-sample jump:  0.1-0.4: 0.37V | 0.3-0.6: 0.47V | 0.7-1.0: 0.71V
exp06 (cutoffs 1.5/1.8), NMC last-sample jump:  0.1-0.4: 1.34V | 0.7-1.0: 1.49V
exp06 (cutoffs 1.5/1.8), LFP last-sample jump:  0.1-0.4: 0.54V | 0.7-1.0: 1.02V
```

**Experiment 03 is contaminated.** Its own `RESULTS.md` reports
`dV_dQ_V_2.4_2.3`/`dV_dQ_V_2.3_2.2` (exactly NMC's 2.3V cutoff) as
surviving bins for its two best-performing intervals -- 0.3-0.6
(97.94%/98.97%) and 0.1-0.4 (94.85%/91.75%) -- and its interpretation
section attributes this to "more shared signal appears at lower SOC...
giving both chemistries a real, comparable presence." Rebuilding those
exact bins from exp03's own raw data and comparing the two chemistries'
actual dV/dQ values (not just whether the bin is covered) confirms this
is the artifact, not signal -- NMC's values are 50-2000x larger in
magnitude than LFP's, consistent with a near-zero final capacity
increment blowing up the dV/dQ ratio at the termination step, not a real
electrochemical signature:

```
exp03, SOC 0.1-0.4, bin dV_dQ_V_2.3_2.2:
  NMC: n=115, mean=-586.5, std=1695.2, range down to -15,714
  LFP: n=80,  mean=-13.6,  std=6.3,    range -5 to -27

exp03, SOC 0.3-0.6, bin dV_dQ_V_2.3_2.2:
  NMC: n=114, mean=-673.4, std=2380.5, range down to -17,833
  LFP: n=51,  mean=-9.9,   std=4.6,    range -4 to -24
```

**Experiment 06 appears clear of this specific issue.** Its lower cutoffs
(1.5V/1.8V) push each chemistry's own artifact bin down to ~1.5-1.9V,
well below its reported surviving-bin range (2.6-3.4V across all four
intervals) -- those artifact bins never make it into 06's trained
features (the other chemistry's coverage there is presumably too sparse
to pass the mutual-coverage filter, though that wasn't separately
verified).

**Not yet done, flagged for a future pass**:
- Confirm/quantify the same magnitude check for exp03's other affected
  bins and re-derive what its accuracy would look like with the artifact
  bin(s) excluded, to see how much of its 91-99% low-SOC accuracy survives
  on genuine signal alone.
- Check whether experiment 04 (pulse-variable, also continuous-adjacent?)
  or experiment 01 have any analogous termination-adjacent artifact, given
  they use the archived pulse protocol rather than this continuous one --
  likely a different mechanism (or none), not checked here.
- Decide on a fix: options include dropping the last N samples per battery
  before binning, excluding any bin within one bin-width of either
  chemistry's cutoff, or flagging near-cutoff bins some other way rather
  than silently trusting the existing `>=20%` mutual-coverage filter to
  catch this (it doesn't -- coverage and value-magnitude are different
  failure modes).
- Re-evaluate experiment 03's RESULTS.md interpretation and any
  conclusions elsewhere in the repo that cite its 0.3-0.6/0.1-0.4 numbers
  as evidence of genuine low-SOC separability.
