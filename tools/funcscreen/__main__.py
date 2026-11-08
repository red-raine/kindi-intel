"""python -m tools.funcscreen --pack PACK [--tokens 200] [--out run.json]
   python -m tools.funcscreen --compare a.json b.json [--out cmp.json]
   python -m tools.funcscreen --selftest
"""
import sys

from tools.funcscreen.score import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
