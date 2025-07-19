#!/bin/bash

# Clear the screen for a clean start
clear

# Initialize Conda for the script's shell session
eval "$(conda shell.bash hook)"

# --- Introduction ---
echo "================================================================="
echo ""
echo "           UniMAG - QE (Quality Evaluation) Demo"
echo ""
echo "================================================================="
echo ""
echo "This script will demonstrate the three core tasks of the QE module"
echo "on the 'Grocery' dataset using the 'GCN' model."
echo ""
read -p "Press any key to start the process..." -n 1 -s

# --- Step 1: Modality Matching ---
clear
echo "-----------------------------------------------------------------"
echo "[STEP 1/4] Running Modality Matching Task..."
echo "-----------------------------------------------------------------"
echo "  [Goal]"
echo "    Evaluate how well image and text modalities are aligned after being"
echo "    enhanced by the graph structure."
echo ""
echo "  [Inputs]"
echo "    - Raw image embeddings (.npy)"
echo "    - Raw text embeddings (.npy)"
echo "    - Graph structure (edge index)"
echo ""
echo "  [Process]"
echo "    1. The GNNTrainer is instantiated."
echo "    2. It checks for a pre-trained GNN model. If not found, it trains one"
echo "       using a contrastive loss (InfoNCE) on the graph data."
echo "    3. The trained GNN enhances the raw embeddings with neighborhood info."
echo "    4. The MatchingEvaluator calculates the CLIP-score for each node's"
echo "       enhanced image-text embedding pair."
echo ""
echo "  [Output]"
echo "    - The average CLIP-score across all nodes in the graph."
echo ""
read -p "Press any key to start..." -n 1 -s
echo ""
conda activate MAGB && python src/main.py task=modality_matching model=gcn dataset=grocery
echo ""
echo "[STEP 1/4] Modality Matching Task COMPLETED."
read -p "Press any key to continue to the next step..." -n 1 -s

# --- Step 2: Modality Retrieval ---
clear
echo "-----------------------------------------------------------------"
echo "[STEP 2/4] Running Modality Retrieval Task..."
echo "-----------------------------------------------------------------"
echo "  [Goal]"
echo "    Evaluate text-to-image and image-to-text retrieval performance."
echo ""
echo "  [Inputs]"
echo "    - Same as the matching task: raw embeddings and graph structure."
echo ""
echo "  [Process]"
echo "    1. (Stage 1) The GNNTrainer provides the GNN-enhanced embeddings."
echo "       (It will load the cached model from the previous step, if not found, it will train a new one)."
echo "    2. (Stage 2) The RetrievalTrainer trains a specialized Two-Tower model"
echo "       on top of these enhanced embeddings to further optimize them for"
echo "       retrieval."
echo "    3. The RetrievalEvaluator uses the final projected embeddings from the"
echo "       Two-Tower model to perform retrieval and calculate metrics."
echo ""
echo "  [Output]"
echo "    - MRR (Mean Reciprocal Rank) and Hits@K for both text-to-image"
echo "      and image-to-text retrieval."
echo ""
read -p "Press any key to start..." -n 1 -s
echo ""
conda activate MAGB && python src/main.py task=modality_retrieval model=gcn dataset=grocery
echo ""
echo "[STEP 2/4] Modality Retrieval Task COMPLETED."
read -p "Press any key to continue to the next step..." -n 1 -s

# --- Step 3: Preprocessing for Alignment ---
clear
echo "-----------------------------------------------------------------"
echo "[STEP 3/4] Preprocessing for Modality Alignment Task..."
echo "-----------------------------------------------------------------"
echo "  [Goal]"
echo "    Generate a cache file with pre-extracted features for the fine-grained"
echo "    alignment task. This is a one-time, offline step."
echo ""
echo "  [Process - Stage 1: Ground Truth Generation]"
echo "    - Input: Raw text and images for each node."
echo "    - Process: For each node, use spaCy to extract noun phrases from text,"
echo "      then use GroundingDINO to find their corresponding bounding boxes"
echo "      in the image."
echo "    - Output: A ground_truth.jsonl file containing (image, text,"
echo "      [(phrase, box), ...]) records."
echo ""
echo "  [Process - Stage 2: Feature Caching]"
echo "    - Input: The ground_truth.jsonl file."
echo "    - Process: For each (phrase, box) pair, extract its text embedding and"
echo "      image region embedding using the specified encoders."
echo "    - Output: An alignment_preprocessed.pt file containing all cached"
echo "      feature pairs."
echo ""
read -p "Press any key to run Stage 1..." -n 1 -s
# echo ""
# conda activate MAGB && python src/multimodal_centric/qe/scripts/prepare_alignment_data.py --dataset grocery --stage 1 --workers-per-gpu 10 10 10 10
echo ""
echo "As Stage 1 requires a lot of time, we will skip it for now. The following STEP 4 will use pre-processed data."
echo "Stage 1 finished."
read -p "Press any key to run Stage 2..." -n 1 -s
# echo ""
# conda activate MAGB && python src/multimodal_centric/qe/scripts/prepare_alignment_data.py --dataset grocery --stage 2 --workers-per-gpu 2 2 2 2
echo ""
echo "As Stage 2 requires a lot of time, we will skip it for now. The following STEP 4 will use pre-processed data."
echo "[STEP 3/4] Preprocessing for Modality Alignment COMPLETED."
read -p "Press any key to continue to the next step..." -n 1 -s

# --- Step 4: Modality Alignment Evaluation ---
clear
echo "-----------------------------------------------------------------"
echo "[STEP 4/4] Running Modality Alignment Evaluation..."
echo "-----------------------------------------------------------------"
echo "  [Goal]"
echo "    Evaluate the fine-grained alignment between textual phrases and"
echo "    specific image regions."
echo ""
echo "  [Inputs]"
echo "    - The alignment_preprocessed.pt file generated in the previous step."
echo "    - The trained GNN model (loaded from cache)."
echo "    - The global feature matrix of the entire graph."
echo ""
echo "  [Process]"
echo "    1. The AlignmentEvaluator loads the cached feature pairs (phrase_embed, region_embed)."
echo "    2. For each pair belonging to a node 'N':"
echo "       a. [FEATURE REPLACEMENT] It takes the global feature matrix of the"
echo "          entire graph and temporarily REPLACES the feature vector of node 'N'"
echo "          with the local feature pair (phrase_embed + region_embed)."
echo "       b. [GNN ENHANCEMENT] It feeds this MODIFIED global feature matrix"
echo "          into the GNN to get enhanced embeddings for all nodes."
echo "       c. [SCORE CALCULATION] It extracts the enhanced local phrase and region"
echo "          embeddings from the GNN's output for node 'N' and calculates"
echo "          their similarity score."
echo ""
echo "  [Output]"
echo "    - The average alignment score across all valid phrase-region pairs."
echo ""
read -p "Press any key to start..." -n 1 -s
echo ""
conda activate MAGB && python src/main.py task=modality_alignment model=gcn dataset=grocery
echo ""
echo "[STEP 4/4] Modality Alignment Evaluation COMPLETED."
echo ""

# --- End ---
echo "-----------------------------------------------------------------"
echo ""
echo "All demonstration tasks have been successfully executed."
echo ""
echo "You can now review the output in the console."
echo ""
read -p "Press any key to exit." -n 1 -s
echo ""