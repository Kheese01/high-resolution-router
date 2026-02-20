from selector.rule_based import RuleBasedSelector
from selector.learned import LearnedSelector

def build_selector(
    mode: str = "rule",
    **kwargs
):
    """
    Selector factory

    Parameters
    ----------
    mode : str
        "rule" | "learned"
    kwargs :
        learned:
            model_path: str
            device: str
        rule:
            (none)

    Returns
    -------
    selector object (callable)
    """

    mode = mode.lower()

    if mode == "rule":
        return RuleBasedSelector()

    elif mode == "learned":
        return LearnedSelector(
            model_path=kwargs.get("model_path"),
            device=kwargs.get("device", "cpu")
        )

    else:
        raise ValueError(f"Unknown selector mode: {mode}")
