"""pyright's diagnostics as an input to the pass.

The pass needs pyright for the one thing an AST cannot see: the declared type that a *function* gave a value. `x =
get_int()` states nothing locally; pyright knows `get_int` returns `int`, and its assignment diagnostic is the only
evidence available. So a diagnostic is consumed in two ways - as a definite negative for the location it lands on, and
as the actual type it names, which then joins the union-find and follows the value's aliases.

Two sources, same shape: a live `pyright --outputjson` run, or a saved output file (`--outputjson > out.json`).
"""
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass

# A diagnostic counts as a type conflict only from the rules that are about an assignment. reportArgumentType and its
# siblings describe a *call*: `os.path.join(x, y)` can complain about an argument on a line that also assigns something,
# and reading that as a contradiction about the assigned name is how a pass like this earns its false positives.
ASSIGN_RULES = {"reportAssignmentType", "reportAttributeAccessIssue", "reportGeneralTypeIssues"}

_GOT_WANT = re.compile(r'(?:Type )?"(?P<got>[^"]+)" is not assignable to (?:declared )?(?:type )?"(?P<want>[^"]+)"')


@dataclass
class Diag:
    file: str
    line: int                       # 1-based, as source coordinates are
    col: int                        # 0-based
    severity: str
    rule: str
    message: str
    got: str = None
    want: str = None

    @property
    def is_conflict(self):
        if self.severity != "error" or self.rule not in ASSIGN_RULES:
            return False
        # `Cannot access attribute "x" for class "Y" / Attribute "x" is unknown` is about a missing declaration -
        # dynamic attributes, monkey-patching - and says nothing about types. Only the form that names both types is
        # evidence about an assignment's type.
        if self.rule == "reportAttributeAccessIssue":
            return bool(self.got and self.want)
        return True


def _interpreter():
    """The command that runs pyright here, or None."""
    for cmd in (["pyright"], [sys.executable, "-m", "pyright"]):
        if shutil.which(cmd[0]):
            return cmd
    return None


def run(paths, timeout=600):
    """(diagnostics, error). A missing pyright is an error string, not an exception: the caller can fall back."""
    cmd = _interpreter()
    if cmd is None:
        return [], "pyright is not on PATH and `python -m pyright` is not installed"
    argv = cmd + ["--outputjson"] + list(paths)
    try:
        p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return [], "cannot run %s (%s)" % (" ".join(argv), e)
    out = p.stdout.decode("utf-8", "replace").strip()
    if not out.startswith("{"):
        tail = (p.stderr.decode("utf-8", "replace") or out).strip().splitlines()[-1:] or [""]
        return [], "pyright produced no JSON (exit %d): %s" % (p.returncode, tail[0][:200])
    try:
        return from_output(json.loads(out)), None
    except (ValueError, KeyError) as e:
        return [], "pyright's JSON could not be read (%s)" % e


def load(path):
    """A saved `pyright --outputjson` file. Raises OSError/ValueError with a readable message."""
    with open(path, encoding="utf-8") as f:
        return from_output(json.load(f))


def from_output(data):
    diags = []
    for d in data.get("generalDiagnostics", []):
        rng = d.get("range") or {}
        start = rng.get("start") or {}
        message = d.get("message") or ""
        m = _GOT_WANT.search(message)
        diags.append(Diag(file=os.path.basename(d.get("file") or ""),
                          line=int(start.get("line", 0)) + 1,
                          col=int(start.get("character", 0)),
                          severity=d.get("severity") or "error",
                          rule=d.get("rule") or "",
                          message=" ".join(message.split()),
                          got=m.group("got") if m else None,
                          want=m.group("want") if m else None))
    return diags
