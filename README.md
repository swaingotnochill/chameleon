# mvp-evals

Simple trace-based evals for agent runs.

Current experiment plan:

- [GAIA self-improve plan](docs/GAIA_SELF_IMPROVE_PLAN.md)

Local smoke test:

```bash
python3 tests/smoke_test.py
```

---

## User Space SDK design

1. Offline Evals

This is basically users running evals against a dataset.
Core usage is running in this in CI test runners. A simple sdk script in a CI
workflow.

Users can see eval results in the CI job itself and add conditions. For
distillation and improvement loop back to the prompt, this needs to be in the
platform for ease(not required as of now).

```
const results = evaris.evaluate(
   dataset, @required
   solvers, // confused if this should be solvers or should be the entire trace
               log. will it be any different?
   scorers, @required
   mode = "offline", // offline | online , @required
)
```

result object -> each sample (input, output, expected output, score object)
score object -> verdict=pass/fail, cost, latency, each trace analysis?? (maybe overkill?), reason


2. Online/Real-Time Evals

This is basically users running evals against real-time production traffic.
Major usecase can be real-time monitoring, and then filtering traces/inputs to
create a reproducible error state, flag issues, and update the offline_evals
dataset. 
Now, can you turn on real-time evals all the time? (for large number of traces,
having llm_as_judge running this will be super expensive.) (maybe flag this on
failed runs? where no desired output?)
Can also be triggered on manual trigger or user feedback loop (not required now.)
Cost = no. of traces * llm_as_judge + k (server cost for processing these evals) 
                                    + no. of traces * average storage cost for each trace
```
const results = evaris.evaluate(
      trace, @required
      scorers, @required
      mode = "online", // offline | online, @required
)
```

result object -> input(entire trace ref?), score
score object -> verdict=anamoly/normal, reason, score object

## Try 1

This is a log of run and comparison of a langchain example with different evals
platform and our mvp.

```
evaluate(dataset: Dataset | None, trace, scorers, mode= Offline | Online)
```

