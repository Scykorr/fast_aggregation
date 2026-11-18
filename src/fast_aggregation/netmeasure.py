"""Перевод объёма и времени в мегабиты в секунду."""


def megabits_per_second(nbytes: int, seconds: float) -> float:
    if nbytes <= 0 or seconds <= 0:
        return 0.0
    return (nbytes * 8) / 1_000_000 / seconds
