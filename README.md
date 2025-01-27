# Signlinformer: From Text to Gesture Translation

Source code for the project **"Signlinformer: From Text to Gesture Translation"**  
*(Ben Saunders, Necati Cihan Camgoz, Richard Bowden - ECCV 2020)*

## Overview
This project implements a system for **text-to-gesture translation** using the **Signlinformer model**, which focuses on low computational complexity while maintaining accurate keypoint generation for sign language production. The system uses progressive transformers to map text inputs to gesture keypoints.

## Data
The project uses the **How2Sign** dataset, which can be downloaded from [How2Sign](https://how2sign.github.io). 

1. **Skeleton Extraction**: Skeleton joints can be extracted using [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose).  
2. **3D Joint Lifting**: Use the **2D to 3D Inverse Kinematics code** available at [SignLanguageProcessing](https://github.com/gopeith/SignLanguageProcessing) under the `3DposeEstimator` directory.  

## Data Preparation
Prepare the How2Sign (or any other sign language dataset) as `.txt` files in the following format:

- **`src` file**: Contains source sentences, with each line representing a new sentence.
- **`trg` file**: Contains skeleton data of each frame, with a space separating frames.  
  - Each frame should contain **150 joint values** (scaled by dividing by 3) and a subsequent counter value, all separated by spaces.
  - Each sequence is separated by a new line.  
  - Ensure that `trg_size` is set to **150** in the configuration file if your data matches this joint count.
- **`files` file**: Contains the name of each sequence, with one name per line.

### Example Data Format
Examples can be found in `/Data/tmp`. Each sequence consists of:

1. **Source Text (`src`)**: `"Hello, how are you?"`  
2. **Target Skeleton Data (`trg`)**: Joint values and counters for each frame in the sequence.  
3. **Sequence Names (`files`)**: `"example_sequence_01"`

## Model
The **Signlinformer** model utilizes Linformer variant of transformers to reduce computational complexity for **text-to-gesture keypoint generation**. The model is trained to generate **keypoints** that represent human gestures in 3D space.

## Key Features
- **Low Computational Complexity**: Optimized for real-time generation of gesture keypoints.  
- **High Accuracy**: Maintains robust mapping from text to gestures.  
- **Video Output**: Generates videos with tracked gestures based on predicted keypoints.

## Usage
1. **Download Data**: Follow the instructions above to download and prepare the dataset.  
2. **Configure Settings**: Specify data paths and set `trg_size` in the config file.  
3. **Run the Model**: Train and evaluate the Signlinformer model on prepared data.  

## Acknowledgements
- **How2Sign Dataset**: [How2Sign](https://how2sign.github.io)  
- **OpenPose for Skeleton Extraction**: [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)  
- **3D Pose Estimation**: [SignLanguageProcessing](https://github.com/gopeith/SignLanguageProcessing)  

For further details or examples, please refer to the paper:  
*Ben Saunders, Necati Cihan Camgoz, Richard Bowden - ECCV 2020*.
