"""A narrowing guard that the next assignment contradicts."""
def f(value):
    if isinstance(value, int):
        alias = value
        alias = "two"
    return alias
