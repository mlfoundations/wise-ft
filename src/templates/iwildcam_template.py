from .utils import append_proper_article, get_plural

iwildcam_template = [
    lambda c: f"a photo of {c}.",
    lambda c: f"{c} in the wild.",
]