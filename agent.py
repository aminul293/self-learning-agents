"""A self-learning agent that gets better at text cleanup tasks.

The agent solves tasks by chaining small operations (lowercase,
strip digits, sort words, and friends). It searches for a chain that
turns the raw text into the expected text. Search order matters: the
budget is small, so trying the right operations first is the whole
game.

After every attempt, a reflection step writes lessons into a memory
file: which operations helped, under which conditions. On later
attempts the agent reads those lessons back and tries the promising
operations first. Pass rate climbs as lessons accumulate. That climb
is the demo, and you can watch it happen.

No API key is needed. An optional LLM reflector can be plugged in
through the SELF_LEARNING_LLM environment variable, but the built-in
rule-based reflector is the default and is fully functional offline.
"""

import json
import os


# ------------------------------------------------------------- operations

def _dedupe_words(s):
    seen, out = set(), []
    for w in s.split():
        if w not in seen:
            seen.add(w)
            out.append(w)
    return " ".join(out)


OPS = {
    "strip_ws": lambda s: s.strip(),
    "collapse_spaces": lambda s: " ".join(s.split()),
    "lower": lambda s: s.lower(),
    "upper": lambda s: s.upper(),
    "title_case": lambda s: s.title(),
    "remove_digits": lambda s: "".join(c for c in s if not c.isdigit()),
    "remove_punct": lambda s: "".join(c for c in s if c.isalnum() or c.isspace()),
    "sort_words": lambda s: " ".join(sorted(s.split())),
    "reverse_words": lambda s: " ".join(reversed(s.split())),
    "dedupe_words": _dedupe_words,
    "snake_case": lambda s: "_".join(s.lower().split()),
}


# ------------------------------------------------------------- task suite

TASKS = [
    ("t01", "  Hello World  ", "hello world"),
    ("t02", "IMPORTANT NOTICE", "important notice"),
    ("t03", "Order 66 Confirmed 2 Times", "order confirmed times"),
    ("t04", "banana apple cherry", "apple banana cherry"),
    ("t05", "one two two three one", "one two three"),
    ("t06", "Hello, World! (2024)", "hello world"),
    ("t07", "the quick brown fox", "THE QUICK BROWN FOX"),
    ("t08", "John Doe Smith", "john_doe_smith"),
    ("t09", "  Multiple   Spaces   Here ", "multiple spaces here"),
    ("t10", "Report #9: Sales UP 40%", "report sales up"),
    ("t11", "zebra mango apple", "apple mango zebra"),
    ("t12", "First Second Third", "Third Second First"),
    ("t13", "Mixed CASE Text 123", "mixed case text"),
    ("t14", "hello world program", "Hello World Program"),
    ("t15", "ALERT: System 9 DOWN!!", "alert system down"),
    ("t16", "dog cat bird dog", "bird cat dog"),
    ("t17", "  Zebra   Apple MANGO ", "apple mango zebra"),
]


def task_features(raw, expected):
    feats = []
    if any(c.isalpha() for c in expected) and expected == expected.lower():
        feats.append("target_lower")
    if any(c.isalpha() for c in expected) and expected == expected.upper():
        feats.append("target_upper")
    if expected == expected.title():
        feats.append("target_title")
    if any(c.isdigit() for c in raw) and not any(c.isdigit() for c in expected):
        feats.append("digits_removed")
    punct = set(".,!#:%()")
    if any(c in punct for c in raw) and not any(c in punct for c in expected):
        feats.append("punct_removed")
    if len(expected.split()) < len(raw.split()):
        feats.append("fewer_words")
    if "_" in expected:
        feats.append("snake_target")
    if "  " in raw and "  " not in expected:
        feats.append("spaces_collapsed")
    if expected.split() == sorted(raw.split()) and raw.split() != sorted(raw.split()):
        feats.append("words_sorted")
    if expected.split() == list(reversed(raw.split())) and len(raw.split()) > 1:
        feats.append("words_reversed")
    return feats


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# ------------------------------------------------------------- memory

class Memory:
    """Lessons: feature -> {operation: score}. Persisted as JSON."""

    def __init__(self, path=None):
        self.path = path
        self.lessons = {}
        if path and os.path.exists(path):
            with open(path) as fh:
                self.lessons = json.load(fh)

    def score(self, features, op):
        return sum(self.lessons.get(f, {}).get(op, 0.0) for f in features)

    def credit(self, features, op, amount):
        for f in features:
            self.lessons.setdefault(f, {})
            self.lessons[f][op] = self.lessons[f].get(op, 0.0) + amount

    def save(self):
        if self.path:
            with open(self.path, "w") as fh:
                json.dump(self.lessons, fh, indent=2, sort_keys=True)


# ------------------------------------------------------------- agent

class Agent:
    def __init__(self, memory, node_budget=45, max_depth=4):
        self.memory = memory
        self.node_budget = node_budget
        self.max_depth = max_depth

    def attempt(self, raw, expected):
        """DFS for an operation chain raw -> expected, ordered by the
        lessons in memory. Returns (solved, path, best_distance)."""
        features = task_features(raw, expected)
        ordered = sorted(OPS, key=lambda op: -self.memory.score(features, op))
        budget = [self.node_budget]
        best = {"dist": edit_distance(raw, expected), "path": []}

        def dfs(state, path, depth):
            if state == expected:
                return path
            if depth == self.max_depth or budget[0] <= 0:
                return None
            for op in ordered:
                if op in path:
                    continue
                budget[0] -= 1
                nxt = OPS[op](state)
                if nxt == state:
                    continue
                d = edit_distance(nxt, expected)
                if d < best["dist"]:
                    best["dist"] = d
                    best["path"] = path + [op]
                found = dfs(nxt, path + [op], depth + 1)
                if found is not None:
                    return found
            return None

        solution = dfs(raw, [], 0)
        if solution is not None:
            return True, solution, 0
        return False, best["path"], best["dist"]


def reflect(memory, raw, expected, solved, path):
    """Write lessons from one attempt.

    A solved task strongly credits every operation in the solution.
    A failed task is analyzed, not just mourned: any operation that
    would have moved the raw text measurably closer to the target gets
    credited, and operations on the closest path found get a weak
    credit. That analysis is what turns a failure into a lesson.
    """
    features = task_features(raw, expected)
    if solved:
        for op in path:
            memory.credit(features, op, 2.0)
        return
    base = edit_distance(raw, expected)
    for name, fn in OPS.items():
        nxt = fn(raw)
        if nxt != raw and edit_distance(nxt, expected) < base:
            memory.credit(features, name, 1.0)
    for op in path:
        memory.credit(features, op, 0.25)


def get_llm_reflector():
    """Optional plug point. Set SELF_LEARNING_LLM to a module path of a
    callable reflect(memory, raw, expected, solved, path) to replace
    the built-in reflector. Returns None when unset, which keeps the
    offline rule-based reflector in charge."""
    target = os.environ.get("SELF_LEARNING_LLM")
    if not target:
        return None
    import importlib
    module_name, _, attr = target.rpartition(".")
    return getattr(importlib.import_module(module_name), attr)


def run_suite(tasks, memory, epochs=4, reflector=None):
    """Run the suite with learning. Returns [cold_baseline, epoch1,
    epoch2, ...] pass rates. The cold baseline runs the suite once with
    a frozen copy of the starting memory and no reflection, so the
    improvement the lessons cause is measured, not assumed."""
    reflector = reflector or get_llm_reflector() or reflect
    probe_agent = Agent(_FrozenMemory(memory))
    baseline = sum(probe_agent.attempt(raw, expected)[0]
                   for _n, raw, expected in tasks) / len(tasks)
    history = [baseline]
    agent = Agent(memory)
    for _ in range(epochs):
        passed = 0
        for _name, raw, expected in tasks:
            solved, path, _dist = agent.attempt(raw, expected)
            reflector(memory, raw, expected, solved, path)
            passed += solved
        memory.save()
        history.append(passed / len(tasks))
    return history


class _FrozenMemory:
    """Read-only view over a Memory, for cold baseline probes."""

    def __init__(self, memory):
        self._lessons = memory.lessons

    def score(self, features, op):
        return sum(self._lessons.get(f, {}).get(op, 0.0) for f in features)

    def credit(self, features, op, amount):
        pass

    def save(self):
        pass
