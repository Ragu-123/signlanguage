import os
import torch
import argparse
import numpy as np
from model import build_model
from helpers import load_config, load_checkpoint
from data import load_data
from constants import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN

class SignLanguagePredictor:
    def __init__(self, config_path, checkpoint_path):
        # Load configuration
        print("Debug: Loading configuration...")
        self.cfg = load_config(config_path)

        # Set random seed for reproducibility
        torch.manual_seed(42)

        # Load data to get vocabularies
        print("Debug: Loading vocabularies...")
        _, _, _, self.src_vocab, self.trg_vocab = load_data(cfg=self.cfg)
        print("Debug: First few entries of source vocabulary (src_vocab):")
        for i, (word, index) in enumerate(self.src_vocab.stoi.items()):
            print(f"{word}: {index}")
            if i >= 9:  # Print only the first 10 entries
                break

        print("\nDebug: First few entries of target vocabulary (trg_vocab):")
        for i, (word, index) in enumerate(self.trg_vocab.stoi.items()):
            print(f"{word}: {index}")
            if i >= 9:  # Print only the first 10 entries
                break

        # Determine device (CPU-only as requested)
        self.device = torch.device("cpu")

        # Build model
        print("Debug: Building model...")
        self.model = build_model(
            cfg=self.cfg,
            src_vocab=self.src_vocab,
            trg_vocab=self.trg_vocab
        )

        # Load checkpoint
        print("Debug: Loading model checkpoint...")
        model_checkpoint = load_checkpoint(checkpoint_path, use_cuda=False)
        self.model.load_state_dict(model_checkpoint["model_state"])

        # Move model to device
        self.model.to(self.device)
        self.model.eval()

        # Configuration parameters
        self.max_output_length = self.cfg["training"].get("max_output_length", 100)
        self.pad_index = self.src_vocab.stoi[PAD_TOKEN]
        self.bos_index = self.src_vocab.stoi[BOS_TOKEN]
        self.eos_index = self.src_vocab.stoi[EOS_TOKEN]

        # Get in_trg_size from the model
        self.in_trg_size = self.model.in_trg_size

    def preprocess_input(self, input_sentence):
        """Preprocess input sentence to model's input format"""
        print(f"Debug: Preprocessing input sentence: {input_sentence}")

        # Tokenize and add special tokens
        tokens = [BOS_TOKEN] + input_sentence.strip().split() + [EOS_TOKEN]
        print(f"Debug: Tokenized input: {tokens}")

        # Convert tokens to indices
        input_indices = [
            self.src_vocab.stoi.get(token, self.src_vocab.stoi[UNK_TOKEN]) for token in tokens
        ]
        print(f"Debug: Input indices: {input_indices}")

        # Create tensor and lengths
        src_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0)  # Add batch dimension
        print(f"Debug: Source tensor shape: {src_tensor.shape}")

        src_lengths = torch.tensor([src_tensor.size(1)])  # Sequence lengths

        # Create source mask for attention
        src_mask = (src_tensor != self.pad_index).unsqueeze(1)

        return src_tensor.to(self.device), src_lengths.to(self.device), src_mask.to(self.device)

    def create_trg_mask(self, seq_len):
        """
        Create a target mask for the transformer decoder
        This mask prevents attending to future tokens
        """
        # Create a square mask of size seq_len x seq_len
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask.unsqueeze(0).unsqueeze(0).to(self.device)

    def extract_tensor(self, tensor_or_tuple):
        """
        Safely extract tensor from a tuple or return the tensor itself
        """
        if isinstance(tensor_or_tuple, tuple):
            # Return the first element if it's a tuple
            return tensor_or_tuple[0] if tensor_or_tuple else None
        return tensor_or_tuple

    def predict_keypoints(self, input_sentence, output_file=None):
        """Predict keypoint sequence for a single input sentence"""
        print(f"Debug: Predicting keypoints for input sentence: {input_sentence}")

        # Preprocess input
        src_tensor, src_lengths, src_mask = self.preprocess_input(input_sentence)

        # Safely extract masks and tensors
        src_tensor = self.extract_tensor(src_tensor)
        src_lengths = self.extract_tensor(src_lengths)
        src_mask = self.extract_tensor(src_mask)

        print(f"Debug: Processed src_mask: {src_mask}")

        print("Debug: Running model inference...")
        # Initialize trg_input with <BOS> token
        trg_input = torch.zeros((1, 1, self.in_trg_size), dtype=torch.float32, device=self.device)
        trg_input[0, 0, 0] = 1.0  # Set first element to 1 to represent <BOS>

        outputs = []
        max_length = 150 # Limit to prevent infinite loop

        # Run greedy decoding step-by-step
        with torch.no_grad():
            # First encode the source sequence
            encoder_output = self.model.encode(
                src=src_tensor, 
                src_length=src_lengths, 
                src_mask=src_mask
            )
            encoder_output = self.extract_tensor(encoder_output)

            for step in range(max_length):
                print(f"Debug: Decoding step {step}")
                
                # Create target mask for current sequence length
                trg_mask = self.create_trg_mask(trg_input.size(1))

                # Decode the target sequence
                try:
                    decoder_output = self.model.decode(
                        encoder_output=encoder_output,
                        src_mask=src_mask, 
                        trg_input=trg_input,
                        trg_mask=trg_mask
                    )
                except Exception as e:
                    print(f"Decode method error: {e}")
                    break

                # Extract the tensor if it's a tuple
                decoder_output = self.extract_tensor(decoder_output)

                # Select the last time step's output
                output = decoder_output[:, -1, :]

                # Store the output
                outputs.append(output.cpu().numpy())

                # Prepare next input
                next_input = torch.zeros((1, 1, self.in_trg_size), dtype=torch.float32, device=self.device)
                next_input[0, 0, :] = torch.from_numpy(output.cpu().numpy())
                trg_input = torch.cat([trg_input, next_input], dim=1)

                # Optional: Add stopping condition 
                if step > max_length - 2:
                    break

        # Convert outputs to numpy
        output_np = np.array(outputs).squeeze()
        print(f"Debug: Output numpy array shape: {output_np.shape}")

        # Optionally save to file
        if output_file:
            print(f"Debug: Saving output to file: {output_file}")
            np.savetxt(output_file, output_np, delimiter=',')

        return output_np

def main():
    parser = argparse.ArgumentParser("Sign Language Keypoint Prediction")
    parser.add_argument("--config", required=True, type=str, 
                        help="Path to configuration file")
    parser.add_argument("--checkpoint", required=True, type=str, 
                        help="Path to model checkpoint")
    parser.add_argument("--input", type=str, 
                        help="Input sentence to predict")
    parser.add_argument("--output", type=str, 
                        help="Output file path for keypoints")

    args = parser.parse_args()

    # Create predictor
    print("Debug: Initializing SignLanguagePredictor...")
    predictor = SignLanguagePredictor(args.config, args.checkpoint)

    # Get input sentence
    if args.input is None:
        input_sentence = input("Enter the input sentence: ")
    else:
        input_sentence = args.input

    print("Debug: Running prediction...")
    # Predict and optionally save
    try:
        output_keypoints = predictor.predict_keypoints(
            input_sentence, 
            output_file=args.output
        )

        print(f"Debug: Predicted keypoint sequence shape: {output_keypoints.shape}")

        if args.output:
            print(f"Keypoints saved to {args.output}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
