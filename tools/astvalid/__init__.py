"""A three-valued validity pass over Python assignments.

`check` walks the AST, joins assignment aliases with a union-find, folds in pyright's assignment diagnostics, and
answers TRUE / FALSE / UNKNOWN per equivalence class of assignments. `score` measures the contradiction half of that
against a small hand-labelled sample. Nothing in the inference engine imports this package.
"""
