import numpy as np
import torch
import math
from torchtext.data import Dataset

from helpers import bpe_postprocess, load_config, get_latest_checkpoint, \
    load_checkpoint, calculate_dtw
from model import build_model, Model
from batch import Batch
from data import load_data, make_data_iter
from constants import UNK_TOKEN, PAD_TOKEN, EOS_TOKEN

def validate_on_data(
    model: Model,
    data: Dataset,
    batch_size: int,
    max_output_length: int,
    eval_metric: str,
    loss_function: torch.nn.Module = None,
    batch_type: str = "sentence",
    eval_type: str = "val",
    BT_model: Optional[Model] = None
):
    """
    Validate the model on a given dataset.
    
    :param model: Trained model for generating predictions.
    :param data: Dataset for validation.
    :param batch_size: Batch size for data iteration.
    :param max_output_length: Maximum length for output generation.
    :param eval_metric: Evaluation metric to be used.
    :param loss_function: Loss function for validation.
    :param batch_type: Batch type, either "sentence" or "tokens".
    :param eval_type: Evaluation type, e.g., "val" or "test".
    :param BT_model: Optional backward translation model for evaluation.
    
    :return: Tuple of validation score, loss, references, hypotheses, inputs, DTW scores, file paths.
    """
    valid_iter = make_data_iter(
        dataset=data, batch_size=batch_size, batch_type=batch_type,
        shuffle=True, train=False
    )

    pad_index = model.src_vocab.stoi[PAD_TOKEN]
    model.eval()
    
    with torch.no_grad():
        valid_hypotheses, valid_references, valid_inputs = [], [], []
        file_paths, all_dtw_scores = [], []
        valid_loss, total_ntokens, total_nseqs = 0, 0, 0

        for batches, valid_batch in enumerate(iter(valid_iter)):
            batch = Batch(
                torch_batch=valid_batch,
                pad_index=pad_index,
                model=model
            )
            targets = batch.trg

            # Calculate loss for batch if applicable
            if loss_function and batch.trg is not None:
                batch_loss, _ = model.get_loss_for_batch(
                    batch, loss_function=loss_function
                )
                valid_loss += batch_loss
                total_ntokens += batch.ntokens
                total_nseqs += batch.nseqs

            # Run model for predictions
            if not model.just_count_in:
                output, attention_scores = model.run_batch(
                    batch=batch,
                    max_output_length=max_output_length
                )

            # If model is configured for future prediction
            if model.future_prediction:
                train_output = torch.cat(
                    (train_output[:, :, :train_output.shape[2] // model.future_prediction],
                     train_output[:, :, -1:]),
                    dim=2
                )
                targets = torch.cat(
                    (targets[:, :, :targets.shape[2] // model.future_prediction],
                     targets[:, :, -1:]),
                    dim=2
                )

            # Handle cases where model only outputs counter
            if model.just_count_in:
                output = train_output

            # Aggregate results for evaluation
            valid_references.extend(targets)
            valid_hypotheses.extend(output)
            file_paths.extend(batch.file_paths)
            valid_inputs.extend([
                [model.src_vocab.itos[batch.src[i][j]] for j in range(len(batch.src[i]))]
                for i in range(len(batch.src))
            ])

            # Compute DTW for batch
            dtw_score = calculate_dtw(targets, output)
            all_dtw_scores.extend(dtw_score)

            # Limit the number of batches evaluated if desired
            if batches == math.ceil(20 / batch_size):
                break

        # Calculate DTW metric
        current_valid_score = np.mean(all_dtw_scores)

    return (
        current_valid_score,
        valid_loss,
        valid_references,
        valid_hypotheses,
        valid_inputs,
        all_dtw_scores,
        file_paths
    )
