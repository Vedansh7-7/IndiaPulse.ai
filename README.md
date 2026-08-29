# IndiaPulse.ai

KPI storytelling and root cause investigation engine.

Accenture Innovation Challenge 2026, Round 2. Problem statement: BusinessIntelligence.ai.

A dashboard shows that a metric dropped. It rarely explains why, or what to do next.
IndiaPulse takes a KPI movement and returns an investigation: whether the movement is
real, where it happened, what caused it, what it was not, and which experiment to run
next. When the evidence cannot pick a winner, it says so.

Every number in the output is computed from data. There are no hard-coded effect sizes.

## Run it

```bash
pip install -r requirements.txt
python download_data.py     # public dataset, 59 MB
python build_demo.py        # runs all four investigations
python -m pytest tests/ -q  # 23 tests
```

Then open `web/index.html`. No server, no build step, no login.

## Demo

Four scenarios, one per decision state. Each is a real slice of the dataset.

| Scenario | Result | Shows |
|---|---|---|
| National | CONFIRMED | Cause found, challenged, survived |
| Santa Catarina | INCONCLUSIVE | Real movement the evidence cannot attribute |
| Goias | NOISE | An alarming dip that is statistically ordinary |
| Para | ARTEFACT | The metric itself is flagged before any cause |

Three of the four are the engine declining to answer.

## How it works

```
KPI alert
   |
Scope 0  Triage        signal or noise. Robust z, control limits, CUSUM.
   |                   Noise stops here, before any agent runs.
Scope 1  Localize      mix vs rate. FDR controlled segment scan.
   |
Scope 2  Investigate   parallel agents, one per dimension:
   |                     Ops            delivery SLA, carrier vs seller leg, dose response
   |                     Voice of Cust  review text, topic lift, discrimination
   |                     Data Integrity is the metric trustworthy
   |                     Market         demand vs local trend, web search context
   |
Scope 3  Adversary     Simpson's paradox, difference in differences with
   |                   parallel trends test, order level reverse causation
   |
Scope 4  Arbiter       rank cause families, separability test, narrate,
                       prescribe the next experiment
```

Three design decisions:

**Cause families.** "Delivery metrics degraded" and "customers complained about delivery"
are not rival explanations. They are one cause with two witnesses. Agents carry a
`cause_family`. Members of a family corroborate, only different families compete. A family
scores at its strongest single line of evidence, since sources inside a family are
correlated.

**Agents cannot win by being confident.** Every verdict returns an effect size with a
confidence interval, a temporal judgement, a specificity score, cited evidence, and
`falsifiable_by`: what observation would kill the hypothesis. Ranking uses those fields.

**The integrity agent argues against the data.** It tries to show the metric is lying.
When it fails, that failure strengthens the other hypotheses. When it succeeds, the report
stops and nothing else is claimed.

## Against the brief

| Requirement | Where |
|---|---|
| Explains in natural language what changed | `arbiter._narrative` |
| Identifies likely root causes | Scope 2, ranked by measured evidence |
| Recommends next steps | `arbiter._prescribe`, with a power calculation |
| Uses structured and unstructured data | Metrics agents plus 41k free text reviews |
| Separates meaningful change from noise | Robust z, control limits, Benjamini-Hochberg |
| Moves from correlation to action | Dose response, DiD with parallel trends, order level check |
| Handles genuine ambiguity | Separability test, INCONCLUSIVE plus next experiment |

The recommendation is a test design, not a platitude:

> Geo split intervention experiment. Causation is already established observationally,
> so an RCT to re-prove it would be waste. The open question is whether a carrier
> intervention recovers the metric. Geo split is correct because carrier capacity is
> assigned geographically, so user level randomisation would contaminate arms.
> n = 757 per arm for 80% power to detect a 5.0pp recovery from an 83.7% base.
> About 6 weeks at observed volume.

## What it found

The event window was discovered by the detector, not specified.

- Weekly review score fell 4.228 to 3.904 over 11 weeks from 2018-01-15. Worst week
  6.4 robust SD below baseline.
- Integrity checks all stable, so the movement is real.
- 99% rate, 0% mix. The same customers had a worse experience.
- On time delivery 93.6% to 83.7%. Delivery time up 4.4 days. The seller handoff moved
  only 0.30 days, so the carrier leg accounts for 93% of the increase.
- Delivery complaints rose 2.3x while product quality complaints fell to 0.91x and wrong
  item stayed at 1.03x. The rise is confined to one theme.
- Demand hypothesis rejected. Volume looks 1.54x elevated against the global baseline, but
  that baseline spans a growth period. Against the prior 8 weeks it is 1.00x, p = 0.975.
- Survived 3 of 3 challenges. DiD -0.556, 95% CI [-0.631, -0.480], parallel pre trends
  p = 0.199. Late orders scored 2.24 against 4.22 for on time orders.

## Data

Public Olist Brazilian E-Commerce dataset. About 99k orders and 41k free text reviews with
order and delivery timestamps. It is used because it pairs a real KPI time series with
timestamped unstructured text, which the brief requires.

The method needs a KPI series plus timestamped text and applies wherever both exist.

## Layout

```
engine/
  config.py        thresholds and topic lexicon
  data.py          load, join, KPI panel, topic flags
  stats_core.py    robust baseline, CUSUM, BH-FDR, bootstrap, Welch, DiD, power
  detect.py        Scope 0
  localize.py      Scope 1
  agents/          Scope 2
  adversary.py     Scope 3
  arbiter.py       Scope 4
  run.py           orchestrator
web/index.html     dashboard
tests/             23 tests
```

## Limitations

- Topic detection is lexicon based, not learned. Transparent but will miss paraphrase.
  Only 41% of orders carry text, so text rates describe reviewers, not all customers.
- Both metrics are cohort anchored to purchase week, so they move at lag 0 by design.
  The engine reports this instead of claiming a lead it cannot show.
- DiD assumes parallel pre trends. The engine tests this and reports the result, but a
  passing test is not proof.
- Retrieved web context can raise a mechanism. It is excluded from evidence scoring.
- Evidence weights are a ranking rule, documented in `agents/base.py`. They order findings
  the data produced. They cannot create one.

Specified but not built: pricing and product funnel agents, live web retrieval, synthetic
control, LLM authored narrative. The narrative is currently templated from computed values,
so the numbers never come from a language model.

## Accessibility

Semantic landmarks, keyboard operable tabs with arrow keys, visible focus, WCAG AA contrast
in both themes, charts carry a description and a table alternative, respects
`prefers-reduced-motion`, no horizontal page scroll.
