# metrics/dtw.py

import os
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import json

def load_keypoints_from_directory(directory, sentence_name):
    """Loads frame-by-frame keypoints for a given sentence from the specified directory."""
    sentence_path = os.path.join(directory, sentence_name)
    frames = sorted(os.listdir(sentence_path))
    keypoints_sequence = []

    for frame in frames:
        with open(os.path.join(sentence_path, frame), 'r') as f:
            keypoints = json.load(f)['people'][0]['pose_keypoints_2d']
            keypoints_sequence.append(keypoints)
    
    return np.array(keypoints_sequence)

def calculate_dtw(predicted_keypoints, target_directory, sentence_name):
    """
    Calculate the DTW distance between predicted keypoints and ground truth keypoints
    loaded from the target directory based on sentence name.
    
    Args:
        predicted_keypoints (np.array): Array of predicted keypoints [seq_len, num_keypoints].
        target_directory (str): Directory containing ground truth keypoint folders.
        sentence_name (str): Name of the sentence corresponding to the ground truth sequence.
        
    Returns:
        float: DTW distance.
    """
    # Load the ground truth keypoints from the target directory
    target_keypoints = load_keypoints_from_directory(target_directory, sentence_name)
    
    # Calculate DTW between predicted and target keypoints
    distance, _ = fastdtw(predicted_keypoints, target_keypoints, dist=euclidean)
    return distance
