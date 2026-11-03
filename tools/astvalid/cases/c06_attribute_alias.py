"""An attribute read into a name, then the attribute rewritten."""
class Box:
    pass

box = Box()
box.n = 1
read = box.n
box.n = "one"
