"""A dict entry rebound to another type: subscripts are not tracked."""
d = {}
d["k"] = 1
d["k"] = "one"
