"""Three-valued truth for the validity pass.

A claim about a piece of code is TRUE when the evidence proves it consistent, FALSE when the evidence proves it
inconsistent, and UNKNOWN when the pass has nothing to say. UNKNOWN is a first-class answer, not an error: it is what
keeps the pass from guessing, and the score reports how often it abstains.

The values are ordered FALSE < UNKNOWN < TRUE so that Kleene's connectives are `min` and `max`:
FALSE and UNKNOWN is FALSE, TRUE and UNKNOWN is UNKNOWN, TRUE or UNKNOWN is TRUE, FALSE or UNKNOWN is UNKNOWN.
"""
from enum import IntEnum


class Tri(IntEnum):
    FALSE = 0
    UNKNOWN = 1
    TRUE = 2

    def __str__(self):
        return self.name.lower()

    @property
    def symbol(self):
        return {Tri.FALSE: "false", Tri.UNKNOWN: "unknown", Tri.TRUE: "true"}[self]


def AND(*values):
    """Kleene conjunction: FALSE if any is FALSE, TRUE if all are TRUE, otherwise UNKNOWN."""
    return Tri(min(int(v) for v in values))


def OR(*values):
    """Kleene disjunction: TRUE if any is TRUE, FALSE if all are FALSE, otherwise UNKNOWN."""
    return Tri(max(int(v) for v in values))


def NOT(value):
    return Tri.UNKNOWN if value is Tri.UNKNOWN else Tri(2 - int(value))


def parse(text):
    """`true`/`false`/`unknown` (any case) -> Tri; anything else is a mistake, not a default."""
    t = str(text).strip().lower()
    if t in ("true", "t", "1"):
        return Tri.TRUE
    if t in ("false", "f", "0"):
        return Tri.FALSE
    if t in ("unknown", "u", "?"):
        return Tri.UNKNOWN
    raise ValueError("not a three-valued constant: %r" % (text,))
