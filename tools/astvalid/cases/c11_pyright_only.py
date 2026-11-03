"""A class attribute's declared type: stated in the class body, not at this assignment."""


class Counter:
    count: int


counter = Counter()
counter.count = "many"
