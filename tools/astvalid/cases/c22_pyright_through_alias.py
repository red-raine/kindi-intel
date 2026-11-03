"""pyright sees the attribute's declared type; the alias makes the finding reach both names."""
class Counter:
    count: int


counter = Counter()
alias = counter.count
counter.count = "many"
