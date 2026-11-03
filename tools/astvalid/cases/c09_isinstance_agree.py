"""A narrowing guard and a literal agree."""
def f(value):
    if isinstance(value, int):
        alias = value
        alias = 2
    return alias
