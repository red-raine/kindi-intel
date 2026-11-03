"""A guard describes the incoming value; assigning replaces it rather than reading it again."""


def normalise(prompt):
    if isinstance(prompt, str):
        prompt = [prompt]
    return prompt
