#!/usr/bin/env bash
# Kindi server setup: the one venv that `kindi serve` needs, from a clean checkout.
#
#   bash setup-server.sh                             # ~/.kindi/server-venv, with fastapi and uvicorn in it
#   bash setup-server.sh --dir /tmp/kindi-venv       # somewhere else
#   bash setup-server.sh --python /usr/bin/python3.12
#
# Only `kindi serve` needs fastapi and uvicorn: chat, verify, convert, bench, bench-spec and the CPU checks are pure
# numpy. The venv is made with --system-site-packages so it sees the numpy that Kindi already runs on, and only
# fastapi and uvicorn go into it. A PEP 668 system (Ubuntu 23.04+, Debian 12+, Fedora 38+) therefore needs neither
# --break-system-packages nor root. Running the script again is cheap, and repairs a half-installed venv.
# Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
DIR="${KINDI_SERVER_VENV:-${KINDI_HOME:-$HOME/.kindi}/server-venv}"
PY="${KINDI_PYTHON:-}"

usage() {
  cat <<EOF
Kindi server setup: makes the venv that \`kindi serve\` needs (fastapi and uvicorn).

  setup-server.sh [options]

Options:
  --dir D          where the venv goes (default ${DIR})
  --python P       the interpreter to build it from (default: python3, then python on PATH)
  -h, --help       this text

Environment: KINDI_SERVER_VENV (the venv), KINDI_PYTHON (the interpreter).
The other Kindi commands need numpy only, so they run under any python that has it.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR="$2"; shift ;;
    --python) PY="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1 (see --help)"; exit 2 ;;
  esac
  shift
done

B=$(printf '\033[1m'); G=$(printf '\033[32m'); Y=$(printf '\033[33m'); R=$(printf '\033[31m'); N=$(printf '\033[0m')
[ -t 1 ] || { B=""; G=""; Y=""; R=""; N=""; }
STEP=0
step() { STEP=$((STEP + 1)); echo; echo "${B}[$STEP] $1${N}"; [ -n "${2:-}" ] && echo "    $2"; return 0; }
info() { echo "    $*"; }
ok() { echo "    ${G}ok${N}  $*"; }
warn() { echo "    ${Y}note${N} $*"; }
die() { echo; echo "${R}stopped:${N} $*"; exit 1; }

# the python inside a venv: bin/ on Linux and macOS, Scripts/ on Windows
venv_python() {
  local p
  for p in "$DIR/bin/python" "$DIR/bin/python3" "$DIR/Scripts/python.exe"; do
    [ -x "$p" ] && { echo "$p"; return 0; }
  done
  return 1
}

echo "${B}Kindi server setup${N}"
info "the repository: ${HERE}"
info "the venv:       ${DIR}"

# ------------------------------------------------------------------------------------------------ 1. python
step "Find a Python" "Kindi needs 3.6 or later. This interpreter builds the venv; the venv keeps its own copies of fastapi and uvicorn."
if [ -n "$PY" ]; then
  command -v "$PY" >/dev/null || die "--python ${PY} is not on PATH"
else
  for c in python3 python; do command -v "$c" >/dev/null && { PY=$c; break; }; done
  [ -n "$PY" ] || die "no python found (a python3 on PATH, or use --python)"
fi
PYV=$("$PY" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 6) else 1)' \
  || die "${PY} is Python ${PYV}; Kindi needs 3.6 or later"
ok "Python ${PYV} (${PY})"
NUMPY=$("$PY" -c 'import numpy; print(numpy.__version__)' 2>/dev/null || true)
if [ -n "$NUMPY" ]; then
  ok "numpy ${NUMPY} (the venv shares this one)"
else
  warn "this python has no numpy: the venv gets its own (Kindi needs it for every command)"
fi

# ------------------------------------------------------------------------------------------------ 2. venv
step "Make the venv" "With --system-site-packages: the venv uses the numpy this python already has."
if VPY=$(venv_python); then
  ok "the venv is already here (${VPY})"
else
  [ -e "$DIR" ] && warn "${DIR} exists without a python in it: the venv is built again"
  mkdir -p "$(dirname "$DIR")"
  "$PY" -m venv --system-site-packages --clear "$DIR" \
    || die "python -m venv failed (on Debian and Ubuntu: apt-get install python3-venv; on Windows: \`py -m pip install virtualenv\`)"
  VPY=$(venv_python) || die "the venv has no python in it (${DIR})"
  ok "created ${DIR}"
fi
if ! "$VPY" -m pip --version >/dev/null 2>&1; then
  info "the venv has no pip: adding it with ensurepip"
  "$VPY" -m ensurepip --upgrade >/dev/null 2>&1 || die "the venv has no pip, and ensurepip did not work"
fi

# ------------------------------------------------------------------------------------------------ 3. packages
step "Install fastapi and uvicorn" "Only the server needs them; they go into the venv, so the system's python is untouched."
PKGS=(fastapi uvicorn)
"$VPY" -c 'import numpy' 2>/dev/null || PKGS+=(numpy)
if "$VPY" -c 'import numpy, fastapi, uvicorn' 2>/dev/null; then
  ok "already installed"
else
  info "\$ ${VPY} -m pip install ${PKGS[*]}"
  "$VPY" -m pip install --no-input --disable-pip-version-check "${PKGS[@]}" \
    || die "pip install failed (no network, or a proxy: set HTTPS_PROXY and run this again)"
  ok "installed"
fi

# ------------------------------------------------------------------------------------------------ 4. check
step "Check that the venv runs the server" "It imports the server modules from this checkout, so a missing file shows here and not at the start of a request."
VOUT=$(cd "$HERE" && "$VPY" - <<'EOF'
import sys
import numpy, fastapi, uvicorn
import kindi.server, kindi.chat, kindi.verify
print("python %s, numpy %s, fastapi %s, uvicorn %s"
      % (sys.version.split()[0], numpy.__version__, fastapi.__version__, uvicorn.__version__))
EOF
) || die "the venv cannot import the server (the error is above)"
ok "${VOUT}"

# ------------------------------------------------------------------------------------------------ done
echo
echo "${G}${B}The server venv is ready.${N}"
echo
echo "  cd ${HERE}"
echo "  ${VPY} -m kindi serve"
echo
info "The other commands (chat, verify, convert, bench) run under any python with numpy: python3 -m kindi verify"
info "The venv does not change the pack, the config or the state: they stay in ${KINDI_HOME:-$HOME/.kindi} (or --pack,"
info "--config, --state-dir)."
