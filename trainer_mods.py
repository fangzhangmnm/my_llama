import torch,logging
from typing import List,Optional

# use stablemax and orthograd to improve grokking
# https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/

def stablemax_cross_entropy_simplified(logits, target, reduction="mean"):
    logits = stablemax_s(logits)
    stablemax = torch.log(logits) - torch.log(torch.sum(logits, dim=-1, keepdim=True))
    if reduction == "mean":
        return -torch.mean(target * stablemax)
    elif reduction == "sum":
        return -torch.sum(target * stablemax)
    else:
        raise RuntimeError()

def stablemax_cross_entropy(logits, labels, reduction="mean"):
    logprobs = log_stablemax(logits, dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
    loss = -torch.mean(prediction_logprobs) if reduction=="mean" else - prediction_logprobs
    return loss

def log_stablemax(x, dim=-1):
    s_x = stablemax_s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))

# def stablemax_s(x, epsilon=1e-30):
def stablemax_s(x, epsilon=1e-6):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )

def labels_to_one_hot(labels, num_classes):
    one_hot = torch.zeros(num_classes, dtype=torch.float64, device=labels.device)
    one_hot[labels] = 1
    return one_hot


def with_orthogonal_gradient(optimizer_class):
    """
    Decorator that wraps an optimizer to perform orthogonal gradient decomposition
    before the standard optimization step. It discards the gradient component
    parallel to the weight vector, preserving only the orthogonal component.
    """
    class OrthogonalOptimizer(optimizer_class):
        def __init__(
            self,
            params,
            skip_orthogonal_param_types: Optional[List[str]] = None,
            skip_orthogonal_1d: bool = True,
            *args,
            **kwargs
        ):
            super().__init__(params, *args, **kwargs)
            self.skip_orthogonal_param_types = skip_orthogonal_param_types or []
            self.skip_orthogonal_1d = skip_orthogonal_1d
            logging.info(
                f"Orthogonal optimizer initialized. "
                f"Skipping param types: {self.skip_orthogonal_param_types}, "
                f"1D params: {self.skip_orthogonal_1d}"
            )

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                loss = closure()

            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    # Optionally skip 1D parameters (bias, layernorm, etc.)
                    if self.skip_orthogonal_1d and p.ndim == 1:
                        logging.debug(f"Skipping orthogonal decomposition for 1D parameter: {getattr(p, 'name', 'unknown')}")
                        continue

                    # Skip based on param name substrings
                    param_name = getattr(p, 'name', None)
                    if param_name and any(skip_type in param_name for skip_type in self.skip_orthogonal_param_types):
                        logging.debug(f"Skipping orthogonal decomposition for parameter: {param_name}")
                        continue

                    # Decompose gradient
                    grad = p.grad
                    norm = torch.linalg.vector_norm(p).clamp_min(1e-9)

                    # If norm is too small, skip
                    if norm < 1e-9:
                        logging.warning(
                            f"Parameter norm is very small ({norm:.8f}), skipping orthogonal decomposition for parameter: {param_name}"
                        )
                        continue

                    normed_weights = p / norm
                    parallel_component = (grad * normed_weights).sum() * normed_weights
                    orthogonal_component = grad - parallel_component
                    p.grad.copy_(orthogonal_component)

                    logging.debug(f"Applied orthogonal decomposition to parameter: {param_name}")

            super().step(closure)
            return loss

    return OrthogonalOptimizer