"""A pyright error about a call, which says nothing about assigned types."""


def takes_nothing():
    return 0


value = takes_nothing(1)
value = value
