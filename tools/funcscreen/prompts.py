"""The 40 fixed prompts the functional screen is scored on (masterplan P2).

Twenty code completions and twenty arithmetic items. Code prompts are prefix text in the style the repository's
prompt sets use - text a trained model continues, not an instruction in a particular format - and each carries the
*declared facts* the screen checks the continuation against: a function name and input/output pairs the completed
function must satisfy. Arithmetic prompts end where a number is expected and carry the expected value.

`sha256()` is written into every run's JSON, for the same reason `tools.kvq.prompts.sha256` exists: a screen whose
items quietly change is a gate that quietly changes, and a paired comparison over items means nothing if the two arms
were scored on different sets.
"""
import hashlib

# (id, kind, prompt, function_name, [(args_tuple, expected), ...]) for code;
# (id, kind, prompt, None, expected_number_string) for arithmetic.
ITEMS = [
    # --- code: 20 completions, each a pure function with declared input/output facts ---
    ("code-01", "code", "def add(a, b):\n    return", "add", [((2, 3), 5), ((-1, 1), 0), ((10, 20), 30)]),
    ("code-02", "code", "def is_even(n):\n    return", "is_even", [((2,), True), ((3,), False), ((0,), True)]),
    ("code-03", "code", "def factorial(n):\n    if n <= 1:\n        return 1\n    return", "factorial",
     [((0,), 1), ((5,), 120), ((3,), 6)]),
    ("code-04", "code", "def reverse_string(s):\n    return", "reverse_string",
     [(("abc",), "cba"), (("",), ""), (("hello",), "olleh")]),
    ("code-05", "code", "def count_vowels(s):\n    count = 0\n    for ch in s.lower():\n        if ch in \"aeiou\":\n            count += 1\n    return", "count_vowels",
     [(("hello",), 2), (("world",), 1), (("rhythm",), 0)]),
    ("code-06", "code", "def max_of_list(values):\n    best = values[0]\n    for v in values[1:]:\n        if v > best:\n            best = v\n    return", "max_of_list",
     [(([1, 9, 3],), 9), (([5],), 5), (([-2, -7, -1],), -1)]),
    ("code-07", "code", "def sum_list(values):\n    total = 0\n    for v in values:\n        total += v\n    return", "sum_list",
     [(([1, 2, 3],), 6), (([],), 0), (([-1, 1],), 0)]),
    ("code-08", "code", "def clamp(x, lo, hi):\n    if x < lo:\n        return", "clamp",
     [((5, 0, 10), 5), ((-3, 0, 10), 0), ((99, 0, 10), 10)]),
    ("code-09", "code", "def celsius_to_fahrenheit(c):\n    return", "celsius_to_fahrenheit",
     [((0.0,), 32.0), ((100.0,), 212.0), ((-40.0,), -40.0)]),
    ("code-10", "code", "def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 1\n    return", "is_prime",
     [((2,), True), ((9,), False), ((7,), True), ((1,), False)]),
    ("code-11", "code", "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return", "gcd",
     [((12, 18), 6), ((7, 13), 1), ((0, 5), 5)]),
    ("code-12", "code", "def word_count(text):\n    return", "word_count",
     [(("hello world",), 2), (("",), 0), (("a b c",), 3)]),
    ("code-13", "code", "def sum_digits(n):\n    total = 0\n    n = abs(n)\n    while n:\n        total += n % 10\n        n //= 10\n    return", "sum_digits",
     [((123,), 6), ((0,), 0), ((999,), 27)]),
    ("code-14", "code", "def power(base, exp):\n    result = 1\n    for _ in range(exp):\n        result *= base\n    return", "power",
     [((2, 3), 8), ((5, 0), 1), ((3, 2), 9)]),
    ("code-15", "code", "def swap(a, b):\n    return", "swap",
     [((1, 2), (2, 1)), (("x", "y"), ("y", "x"))]),
    ("code-16", "code", "def absolute(x):\n    if x < 0:\n        return", "absolute",
     [((-5,), 5), ((3,), 3), ((0,), 0)]),
    ("code-17", "code", "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return", "fibonacci",
     [((0,), 0), ((6,), 8), ((10,), 55)]),
    ("code-18", "code", "def average(values):\n    if not values:\n        return 0\n    return", "average",
     [(([2, 4, 6],), 4.0), (([1, 2],), 1.5), (([10],), 10.0)]),
    ("code-19", "code", "def flatten(groups):\n    out = []\n    for group in groups:\n        out.extend(group)\n    return", "flatten",
     [(([1, 2], [3]), [1, 2, 3]), (([],), []), (([1], [2], [3, 4]), [1, 2, 3, 4])]),
    ("code-20", "code", "def capitalize_first(s):\n    if not s:\n        return s\n    return", "capitalize_first",
     [(("hello",), "Hello"), (("",), ""), (("a",), "A")]),

    # --- arithmetic: 20 exact-match items ---
    ("arith-01", "arith", "If 3x + 7 = 22, then 3x = 15 and x =", None, "5"),
    ("arith-02", "arith", "A train travels 180 kilometres in 2.5 hours, so its average speed is", None, "72"),
    ("arith-03", "arith", "17 + 25 =", None, "42"),
    ("arith-04", "arith", "9 * 8 =", None, "72"),
    ("arith-05", "arith", "100 - 37 =", None, "63"),
    ("arith-06", "arith", "144 / 12 =", None, "12"),
    ("arith-07", "arith", "7 squared plus 5 squared is", None, "74"),
    ("arith-08", "arith", "The next number in the sequence 2, 6, 12, 20, 30 is", None, "42"),
    ("arith-09", "arith", "15% of 200 is", None, "30"),
    ("arith-10", "arith", "2 to the power of 10 is", None, "1024"),
    ("arith-11", "arith", "The sum of the first ten positive integers is", None, "55"),
    ("arith-12", "arith", "If a rectangle has length 9 and width 4, its area is", None, "36"),
    ("arith-13", "arith", "Half of 86 is", None, "43"),
    ("arith-14", "arith", "11 * 11 =", None, "121"),
    ("arith-15", "arith", "50 minus 8 minus 9 is", None, "33"),
    ("arith-16", "arith", "A hexagon has 6 sides, so it has", None, "6"),
    ("arith-17", "arith", "23 + 48 + 29 =", None, "100"),
    ("arith-18", "arith", "6 * 7 * 2 =", None, "84"),
    ("arith-19", "arith", "The square root of 169 is", None, "13"),
    ("arith-20", "arith", "If y = 4x - 3, then when x = 5, y =", None, "17"),
]

CODE = [it for it in ITEMS if it[1] == "code"]
ARITH = [it for it in ITEMS if it[1] == "arith"]


def sha256():
    """digest of the item set as scored: ids, kinds, prompts and declared facts"""
    h = hashlib.sha256()
    for item in ITEMS:
        h.update(repr(item).encode("utf-8") + b"\x00")
    return h.hexdigest()
