"""The same local name in two functions is two locations."""


def first():
    item = 1
    return item


def second():
    item = "one"
    return item
