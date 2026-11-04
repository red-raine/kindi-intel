"""Exodus-Ternera: the parts of the specification that can be executed, and what executing them found.

    python -m tools.exodus               # all three modules, 37 checks
    python -m tools.exodus.rotate        # the rotation stage
    python -m tools.exodus.router        # routing, halting, distillation loss
    python -m tools.exodus.heads         # attention-head slicing

The specification asks for a five-phase compiler that turns frontier MoEs into looped ternary students. That needs donor
checkpoints (Maple/Bonsai, Qwen-2.5-Coder-32B, Ornith-35B-A3B), GPUs and hours of training, and none of it is in this
repository. What is here is the arithmetic core of the three Python modules the document supplies, checked on 64x256
tensors and 4-head attention instead of argued about in prose. Each module's docstring states its findings; each test
pins one down so it cannot quietly come back.
"""
import sys

from . import heads, rotate, router


def main(argv):
    total = 0
    failed = []
    for mod in (rotate, router, heads):
        print("== %s" % mod.__name__.split(".")[-1])
        if mod.selftest() != 0:
            failed.append(mod.__name__)
        total += 1
        print()
    print("%d check groups passed%s" % (total - len(failed), "" if not failed else ", failed: %s" % failed))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
