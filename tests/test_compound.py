# prolog/tests/test_compound.py
from prolog.solver import Solver

def test_compound_single_solution():
    solver = Solver("""
симптом(кашель, грипп).
симптом(температура, грипп).
симптом(головная_боль, мигрень).
симптом(тошнота, мигрень).
симптом(кашель, бронхит).
симптом(одышка, бронхит).
""")
    raw = solver.find_solutions(
        "симптом(кашель, X), симптом(температура, X)."
    )
    # solver отдаёт ровно одно решение, X = грипп
    assert len(raw) == 1
    assert str(raw[0]["X"]) == "грипп"


def test_display_no_phantom_true():
    from prolog.solver import ParseSolution
    q = "симптом(температура, X)."
    raw = [{"X": "грипп"}]    # type: ignore[assignment]
    parsed = ParseSolution(q, raw)
    assert parsed["count"] == 1
    # ключевое: никакого "true" в выводе
    assert parsed["output"] == "X = грипп"