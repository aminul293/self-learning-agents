"""Demo: watch the agent improve as its lessons accumulate.

Starts with an empty memory, runs the built-in task suite for five
epochs, and prints the pass rate after each one. Lessons persist to
lessons.json, so running it again starts from what it already learned.
Delete lessons.json to watch it learn from zero again.
"""

import os

from agent import TASKS, Memory, run_suite

MEMORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lessons.json")


def main():
    fresh = not os.path.exists(MEMORY_PATH)
    memory = Memory(MEMORY_PATH)
    print(f"tasks: {len(TASKS)}   starting memory: {'empty' if fresh else 'loaded from lessons.json'}")
    print("first a cold probe with no learning, then 5 epochs of attempt, reflect, retry...\n")
    history = run_suite(TASKS, memory, epochs=5)
    print(f"cold, no lessons: pass rate {history[0]:5.0%}  {'#' * int(history[0] * 30)}")
    for i, rate in enumerate(history[1:], 1):
        bar = "#" * int(rate * 30)
        print(f"epoch {i}:          pass rate {rate:5.0%}  {bar}")
    gained = history[-1] - history[0]
    print(f"\nimprovement from lessons: {gained:+.0%}")
    print(f"lessons stored: {sum(len(v) for v in memory.lessons.values())} "
          f"across {len(memory.lessons)} conditions in lessons.json")


if __name__ == "__main__":
    main()
