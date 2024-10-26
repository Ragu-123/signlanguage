from torch import nn, Tensor

class RegLoss(nn.Module):
    """
    Regression Loss for pose prediction.
    """

    def __init__(self, cfg, target_pad=0.0):
        super(RegLoss, self).__init__()

        # Select the loss function based on the configuration
        self.loss = cfg["training"]["loss"].lower()
        if self.loss == "l1":
            self.criterion = nn.L1Loss()
        elif self.loss == "mse":
            self.criterion = nn.MSELoss()
        else:
            print("Loss type not found; defaulting to L1 loss.")
            self.criterion = nn.L1Loss()

        # Retrieve loss scale and target padding from configuration
        model_cfg = cfg["model"]
        self.target_pad = target_pad
        self.loss_scale = model_cfg.get("loss_scale", 1.0)

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        """
        Compute the regression loss over the masked predictions.
        """
        # Create a mask for padding
        loss_mask = (targets != self.target_pad)

        # Mask predictions and targets
        preds_masked = preds * loss_mask
        targets_masked = targets * loss_mask

        # Calculate loss over the masked predictions
        loss = self.criterion(preds_masked, targets_masked)

        # Scale the loss if necessary
        if self.loss_scale != 1.0:
            loss *= self.loss_scale

        return loss


class XentLoss(nn.Module):
    """
    Cross-Entropy Loss with optional label smoothing for token prediction.
    """

    def __init__(self, pad_index: int, smoothing: float = 0.0):
        super(XentLoss, self).__init__()
        self.smoothing = smoothing
        self.pad_index = pad_index
        # Cross-entropy with ignored padding tokens
        self.criterion = nn.NLLLoss(ignore_index=self.pad_index, reduction="sum")

    def forward(self, log_probs: Tensor, targets: Tensor) -> Tensor:
        """
        Compute cross-entropy loss between log probabilities and target indices.
        """
        # Flatten target tensor for computing loss
        targets = targets.contiguous().view(-1)

        # Compute loss over log probabilities and target indices
        loss = self.criterion(
            log_probs.contiguous().view(-1, log_probs.size(-1)), targets
        )

        return loss
