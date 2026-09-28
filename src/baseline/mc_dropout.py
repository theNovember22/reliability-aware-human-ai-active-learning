import torch
from torch import nn


def enable_dropout(model):
    """
    Put the model in evaluation mode while keeping
    dropout layers active.
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
    Perform MC Dropout inference.

    Returns:
        mean_probs
        variance
        predictive_entropy
        mutual_information
        predictions
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

    # ------------------------------------------------
    # Mean predictive distribution
    # ------------------------------------------------

    mean_probs = predictions.mean(dim=0)

    # ------------------------------------------------
    # Predictive variance
    # ------------------------------------------------

    variance = predictions.var(
        dim=0,
        unbiased=False
    )

    # ------------------------------------------------
    # Predictive entropy
    # ------------------------------------------------

    eps = 1e-8

    predictive_entropy = -torch.sum(
        mean_probs * torch.log(mean_probs + eps),
        dim=-1
    )

    # ------------------------------------------------
    # Expected entropy
    # ------------------------------------------------

    sample_entropy = -torch.sum(
        predictions *
        torch.log(predictions + eps),
        dim=-1
    )

    expected_entropy = sample_entropy.mean(
        dim=0
    )

    # ------------------------------------------------
    # Mutual information
    # ------------------------------------------------

    mutual_information = (
        predictive_entropy
        - expected_entropy
    )

    return (
        mean_probs,
        variance,
        predictive_entropy,
        mutual_information,
        predictions
    )