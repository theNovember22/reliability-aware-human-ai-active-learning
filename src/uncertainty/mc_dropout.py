import torch
from torch import nn


def enable_dropout(model):
    """
    Enable dropout layers during inference.

    The model itself can remain in eval mode while
    dropout layers are explicitly switched to train mode.
    """
    model.eval()

    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_predict(
    model,
    human_labels,
    features,
    num_samples=20
):
    """
    Perform stochastic forward passes using MC Dropout.

    Returns:
        mean_probs:
            Mean prediction probability across MC samples.

        variance:
            Predictive variance across MC samples.

        predictions:
            All stochastic predictions.
    """

    enable_dropout(model)

    predictions = []

    for _ in range(num_samples):

        probs = model(
            human_labels,
            feature_input=features
        )

        predictions.append(probs)

    predictions = torch.stack(
        predictions,
        dim=0
    )

    mean_probs = predictions.mean(
        dim=0
    )

    variance = predictions.var(
        dim=0
    )

    return (
        mean_probs,
        variance,
        predictions
    )