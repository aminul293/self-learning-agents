# self-learning-agents

An agent that gets measurably better at a task the more it attempts
it, using the two ingredients behind most agent memory systems: a
store of lessons from past attempts, and a reflection step that writes
new lessons after each one.

The built-in job is text cleanup: turn messy strings into clean target
strings by chaining small operations (lowercase, remove digits, sort
words, snake_case, and friends). The interesting part is not the job.
It is the loop.

## Run it

```bash
python demo.py        # watch the pass rate climb over 5 epochs
python -m pytest      # test suite
```

Delete lessons.json and run the demo again to watch it learn from zero.

## Example output

```text
tasks: 17   starting memory: empty
first a cold probe with no learning, then 5 epochs of attempt, reflect, retry...

cold, no lessons: pass rate   71%  #####################
epoch 1:          pass rate   94%  ############################
epoch 2:          pass rate  100%  ##############################
...
improvement from lessons: +29%
```

The demo measures a cold baseline first (same agent, lessons frozen,
no reflection), so the climb is the measured effect of the lessons,
not a vibe.

## How it works

- **Attempts.** For each task the agent depth-first searches for a
  chain of operations that turns the raw text into the expected text.
  It works under a small node budget, so the order it tries operations
  in decides whether it finds the chain in time.
- **Lessons.** Memory is a JSON file mapping conditions about a task
  (target is lowercase, digits were removed, words got sorted) to
  scores per operation. High score means "this operation paid off under
  this condition before."
- **Reflection.** After each attempt, solved tasks strongly credit the
  operations in the solution under the task's conditions. Failed tasks
  are analyzed: any operation that would have moved the text measurably
  closer to the target earns credit too. Failures become lessons, not
  just misses.
- **Retrieval.** On the next attempt, operations are ordered by the
  lessons matching the new task's conditions. Experience literally
  reorders the search.

## Plugging in an LLM

The reflector is a plug point. Set SELF_LEARNING_LLM to a dotted path
of a callable with the signature
reflect(memory, raw, expected, solved, path) and it replaces the
built-in rule-based reflector, for example to have a model write the
lessons in natural language. Unset, everything runs offline with no
API key, which is how the demo and tests run.

## What it demonstrates

Agent memory design, retrieval-conditioned behavior, reflection loops,
and the discipline of measuring whether "learning" actually happened
instead of just claiming it in the README.

## Honest scope

The task suite is small and synthetic, the operations are string
functions, and the reflection is rule-based by default. This is a
clear, runnable model of the loop used by bigger systems, not a
research result.
