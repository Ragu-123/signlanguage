import os
import torch
import argparse
import shutil
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler

from models.sign_language_transformer import SignLanguageTransformer  # Your model
from data.data_loader import SignLanguageDataset  # Your dataset loader
from utils.loss import get_loss_function  # Define your loss function
from utils.metrics import calculate_metrics  # Function for evaluation metrics like DTW
from utils.utils import set_seed, save_checkpoint, load_checkpoint, log_training_info  # Helper utilities
from data.data_config import load_config  # Load your `base.yaml` and `data.yaml` configuration files

class Trainer:
    def __init__(self, model, optimizer, scheduler, train_loader, val_loader, config, logger, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger
        self.device = device

        self.global_step = 0
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        self.early_stop_patience = config['training']['early_stop_patience']
        self.ckpt_dir = config['training']['ckpt_dir']

        # TensorBoard writer
        self.tb_writer = SummaryWriter(log_dir=self.ckpt_dir)

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        for batch in self.train_loader:
            self.global_step += 1

            inputs = batch['inputs'].to(self.device)
            keypoints = batch['keypoints'].to(self.device)

            outputs = self.model(inputs)

            loss = get_loss_function(outputs, keypoints)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()

            if self.global_step % self.config['training']['log_steps'] == 0:
                self.logger.info(f"Step [{self.global_step}] Loss: {loss.item():.4f}")
                self.tb_writer.add_scalar("train/loss", loss.item(), self.global_step)

        avg_loss = total_loss / len(self.train_loader)
        self.logger.info(f"Epoch [{epoch}] Training Loss: {avg_loss:.4f}")

    def validate(self, epoch):
        self.model.eval()
        total_val_loss = 0
        val_score = 0
        val_dtw_score = 0
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch['inputs'].to(self.device)
                keypoints = batch['keypoints'].to(self.device)

                outputs = self.model(inputs)

                # Compute loss and metrics
                loss = get_loss_function(outputs, keypoints)
                total_val_loss += loss.item()

                # Use a custom metric for evaluation (DTW, etc.)
                score, dtw_score = calculate_metrics(outputs, keypoints)
                val_score += score
                val_dtw_score += dtw_score

        avg_val_loss = total_val_loss / len(self.val_loader)
        avg_val_score = val_score / len(self.val_loader)
        avg_dtw_score = val_dtw_score / len(self.val_loader)

        self.logger.info(f"Epoch [{epoch}] Validation Loss: {avg_val_loss:.4f}, DTW Score: {avg_dtw_score:.4f}")
        self.tb_writer.add_scalar("valid/loss", avg_val_loss, epoch)
        self.tb_writer.add_scalar("valid/dtw_score", avg_dtw_score, epoch)

        return avg_val_loss, avg_val_score

    def save_model_checkpoint(self, epoch, val_loss):
        # Save the model if the validation loss is the best
        if val_loss < self.best_val_loss:
            self.logger.info(f"Saving best model at epoch {epoch}")
            save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, val_loss, self.ckpt_dir)
            self.best_val_loss = val_loss
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1

    def train(self):
        num_epochs = self.config['training']['num_epochs']

        for epoch in range(1, num_epochs + 1):
            self.logger.info(f"Starting epoch {epoch}/{num_epochs}")
            self.train_one_epoch(epoch)
            val_loss, val_score = self.validate(epoch)

            # Save model checkpoint
            self.save_model_checkpoint(epoch, val_loss)

            # Early stopping
            if self.early_stop_counter >= self.early_stop_patience:
                self.logger.info("Early stopping triggered.")
                break

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

        self.tb_writer.close()

def main(config_path, checkpoint_path=None):
    # Load config
    config = load_config(config_path)

    # Set random seed
    set_seed(config['training'].get('random_seed', 42))

    # Prepare device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datasets
    train_dataset = SignLanguageDataset(config, mode='train')
    val_dataset = SignLanguageDataset(config, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # Build model
    model = SignLanguageTransformer(config).to(device)

    # Load checkpoint if specified
    if checkpoint_path:
        model, optimizer, scheduler, epoch = load_checkpoint(checkpoint_path, model)

    # Set up optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=config['training']['lr'])
    num_training_steps = len(train_loader) * config['training']['num_epochs']
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Set up logging
    log_dir = config['training']['ckpt_dir']
    logger = log_training_info(log_dir)

    # Initialize trainer
    trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler, train_loader=train_loader,
                      val_loader=val_loader, config=config, logger=logger, device=device)

    # Start training
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Sign Language Transformer model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint file to resume training.")
    args = parser.parse_args()

    main(config_path=args.config, checkpoint_path=args.checkpoint)
