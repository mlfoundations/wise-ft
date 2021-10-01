from .utils import append_proper_article, get_plural

fmow_template = [
    lambda c : f"satellite photo of a {c}.",
    lambda c : f"aerial photo of a {c}.",
    lambda c : f"satellite photo of {append_proper_article(c)}.",
    lambda c : f"aerial photo of {append_proper_article(c)}.",
    lambda c : f"satellite photo of a {c} in asia.",
    lambda c : f"aerial photo of a {c} in asia.",
    lambda c : f"satellite photo of a {c} in africa.",
    lambda c : f"aerial photo of a {c} in africa.",
    lambda c : f"satellite photo of a {c} in the americas.",
    lambda c : f"aerial photo of a {c} in the americas.",
    lambda c : f"satellite photo of a {c} in europe.",
    lambda c : f"aerial photo of a {c} in europe.",
    lambda c : f"satellite photo of a {c} in oceania.",
    lambda c : f"aerial photo of a {c} in oceania.",
    lambda c: f"a photo of a {c}.",
    lambda c: f"{c}.",
]
