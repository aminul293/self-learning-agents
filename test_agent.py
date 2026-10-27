import os

from agent import OPS, Memory, TASKS, edit_distance, run_suite, task_features


def test_ops_are_correct():
    assert OPS["lower"]("HeLLo") == "hello"
    assert OPS["remove_digits"]("a1b22c") == "abc"
    assert OPS["sort_words"]("banana apple") == "apple banana"
    assert OPS["snake_case"]("John Doe") == "john_doe"
    assert OPS["dedupe_words"]("a b a c b") == "a b c"


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("same", "same") == 0
    assert edit_distance("", "abc") == 3


def test_features_detected():
    feats = task_features("Order 66", "order")
    assert "target_lower" in feats
    assert "digits_removed" in feats
    assert "fewer_words" in feats


def test_memory_roundtrip(tmp_path):
    path = str(tmp_path / "lessons.json")
    m = Memory(path)
    m.credit(["target_lower"], "lower", 2.0)
    m.save()
    m2 = Memory(path)
    assert m2.score(["target_lower"], "lower") == 2.0
    assert m2.score(["target_lower"], "upper") == 0.0


def test_agent_improves_with_lessons(tmp_path):
    path = str(tmp_path / "lessons.json")
    memory = Memory(path)
    history = run_suite(TASKS, memory, epochs=5)
    cold, learned = history[0], history[-1]
    assert learned > cold, f"lessons did not help: {history}"
    assert learned >= 0.9, f"final pass rate too low: {history}"


def test_memory_file_is_written(tmp_path):
    path = str(tmp_path / "lessons.json")
    run_suite(TASKS, Memory(path), epochs=1)
    assert os.path.exists(path)
