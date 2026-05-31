"""Reward implementations are intentionally omitted from the anonymous release."""


def default_compute_score(*args, **kwargs):
    raise NotImplementedError(
        "Reward and evaluation code is omitted from this anonymous release. "
        "Provide a private verifier via custom_reward_function.path."
    )


_default_compute_score = default_compute_score

__all__ = ["default_compute_score"]
