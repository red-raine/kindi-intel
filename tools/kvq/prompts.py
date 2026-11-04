"""The 48 prompts the K/V cache formats are compared on, fixed here so the sweep is one number.

The V-format study in `docs/CPU-PERF-2026.md` reports every arm over "48 prompts - factual answers, code in four
languages, maths, translation, JSON, SQL, shell, enumeration, sentiment" and 160 greedy tokens. Those prompts lived in
a shell harness outside the repository, so the one comparison that had no interval behind it could not be re-run from
the tree. This is that set: twelve categories, four prompts each, prompt-only text that asks a trained model to
continue rather than to answer in a particular format.

`sha256()` is written into every sweep's JSON, because a prompt set that quietly changes is a measurement that quietly
changes, and a paired interval over prompts means nothing if the prompts are not the ones that were scored.
"""
import hashlib

# (id, category, text). The id is the pairing key in the bootstrap: one prompt is one observation.
PROMPTS = [
    ("factual-1", "factual", "The capital of France is Paris, and the river that flows through the city is"),
    ("factual-2", "factual", "Water boils at one hundred degrees Celsius at sea level because the vapour pressure"),
    ("factual-3", "factual", "The element with atomic number 26 is iron, and its symbol on the periodic table is"),
    ("factual-4", "factual", "The first person to walk on the Moon was Neil Armstrong, in July of the year"),

    ("python-1", "code-python", "def moving_average(values, window):\n    \"\"\"Return the mean of each trailing window.\"\"\""),
    ("python-2", "code-python", "import json\n\ndef load_config(path):\n    with open(path) as handle:\n        return"),
    ("python-3", "code-python", "class RingBuffer:\n    def __init__(self, size):\n        self.size = size\n        self.items ="),
    ("python-4", "code-python", "for row in rows:\n    if row.get(\"status\") == \"failed\":\n        print(row[\"id\"],"),

    ("javascript-1", "code-javascript", "const debounce = (fn, wait) => {\n  let timer = null;\n  return (...args) => {"),
    ("javascript-2", "code-javascript", "async function fetchJson(url) {\n  const response = await fetch(url);\n  if (!response.ok)"),
    ("javascript-3", "code-javascript", "export default function formatBytes(bytes) {\n  const units = [\"B\", \"KB\", \"MB\", \"GB\"];\n"),
    ("javascript-4", "code-javascript", "document.querySelectorAll(\"button.act\").forEach((button) => {\n  button.addEventListener(\"click\", (event) => {"),

    ("rust-1", "code-rust", "fn longest<'a>(a: &'a str, b: &'a str) -> &'a str {\n    if a.len() > b.len() {"),
    ("rust-2", "code-rust", "struct Point { x: f64, y: f64 }\n\nimpl Point {\n    fn distance(&self, other: &Point) -> f64 {"),
    ("rust-3", "code-rust", "fn main() {\n    let values: Vec<i32> = (1..10).filter(|n| n % 2 == 0).collect();\n    println!(\"{"),
    ("rust-4", "code-rust", "use std::collections::HashMap;\n\nfn counts(words: &[&str]) -> HashMap<&str, usize> {"),

    ("c-1", "code-c", "#include <stdio.h>\n\nint main(void) {\n    for (int i = 0; i < 10; ++i) {"),
    ("c-2", "code-c", "size_t ring_push(ring_t *r, int value) {\n    r->items[r->head] = value;\n    r->head ="),
    ("c-3", "code-c", "static int compare_ints(const void *a, const void *b) {\n    const int x = *(const int *)a;\n"),
    ("c-4", "code-c", "char *dup_string(const char *s) {\n    const size_t n = strlen(s) + 1;\n    char *copy ="),

    ("maths-1", "maths", "A train travels 180 kilometres in 2.5 hours, so its average speed in kilometres per hour is"),
    ("maths-2", "maths", "The sum of the interior angles of a hexagon is 720 degrees, because a hexagon can be split into"),
    ("maths-3", "maths", "If 3x + 7 = 22, then 3x = 15 and x ="),
    ("maths-4", "maths", "The next number in the sequence 2, 6, 12, 20, 30 is 42, because the differences are"),

    ("translation-1", "translation", "Translate into French: the meeting has been moved to Thursday afternoon, and the room is"),
    ("translation-2", "translation", "Translate into German: please close the window before you leave, because the heating"),
    ("translation-3", "translation", "Translate into Spanish: we need three more chairs for the guests who are arriving"),
    ("translation-4", "translation", "Translate into Italian: the invoice was sent last week, but the payment has not"),

    ("json-1", "json", "{\"user\": {\"id\": 42, \"name\": \"Ada\", \"roles\": [\"admin\", \"billing\"], \"active\":"),
    ("json-2", "json", "{\"event\": \"deploy\", \"environment\": \"staging\", \"commit\": \"a91f3c2\", \"duration_ms\":"),
    ("json-3", "json", "[{\"sku\": \"A-100\", \"quantity\": 2, \"price\": 19.5}, {\"sku\": \"B-240\", \"quantity\":"),
    ("json-4", "json", "{\"status\": \"error\", \"code\": 429, \"message\": \"rate limit exceeded\", \"retry_after\":"),

    ("sql-1", "sql", "SELECT customer_id, COUNT(*) AS orders, SUM(total) AS revenue FROM orders WHERE created_at >="),
    ("sql-2", "sql", "WITH ranked AS (SELECT id, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY created_at DESC) AS rn"),
    ("sql-3", "sql", "CREATE TABLE events (id BIGSERIAL PRIMARY KEY, kind TEXT NOT NULL, payload JSONB,"),
    ("sql-4", "sql", "UPDATE inventory SET quantity = quantity - 1 WHERE sku = 'A-100' AND quantity >"),

    ("shell-1", "shell", "find . -name '*.log' -mtime +7 -print0 | xargs -0"),
    ("shell-2", "shell", "du -sh */ 2>/dev/null | sort -h | tail -n"),
    ("shell-3", "shell", "for f in *.png; do convert \"$f\" -resize 50%"),
    ("shell-4", "shell", "tar -czf backup-$(date +%F).tar.gz --exclude='.git'"),

    ("enumeration-1", "enumeration", "The planets of the solar system in order from the Sun are Mercury, Venus, Earth,"),
    ("enumeration-2", "enumeration", "The first eight prime numbers are 2, 3, 5, 7, 11, 13,"),
    ("enumeration-3", "enumeration", "The days of the week in order, starting from Monday, are Monday, Tuesday,"),
    ("enumeration-4", "enumeration", "The primary colours of light are red, green and"),

    ("sentiment-1", "sentiment", "Review: \"The battery lasts two days and the screen is sharp, but the keyboard feels cheap.\" Sentiment:"),
    ("sentiment-2", "sentiment", "Review: \"It arrived three weeks late, the box was crushed, and support never answered.\" Sentiment:"),
    ("sentiment-3", "sentiment", "Review: \"Exactly what the listing described, shipped the same day, and it just works.\" Sentiment:"),
    ("sentiment-4", "sentiment", "Review: \"The first unit was faulty, the replacement works, so I am not sure how to feel.\" Sentiment:"),
]

CATEGORIES = ("factual", "code-python", "code-javascript", "code-rust", "code-c", "maths", "translation", "json",
              "sql", "shell", "enumeration", "sentiment")


def texts():
    """the prompt strings in order, which is the order the sweep reports them in"""
    return [t for _i, _c, t in PROMPTS]


def sha256():
    """digest of the prompt set as scored: ids and texts, categories excluded, so a rename is not a new measurement"""
    h = hashlib.sha256()
    for i, _c, t in PROMPTS:
        h.update((i + "\x00" + t + "\x00").encode("utf-8"))
    return h.hexdigest()


def by_category():
    """category -> how many prompts it has, which the report quotes to show the set is not one kind of text"""
    out = {}
    for _i, c, _t in PROMPTS:
        out[c] = out.get(c, 0) + 1
    return out
