# GAT Model Training on Cora Dataset

IN CONTINUATION OF THE CORA DATASET

This project trains a Graph Attention Network (GAT) model on the Cora dataset for node classification. The model achieves high accuracy, and the performance is evaluated using both training and validation loss, as well as validation accuracy. Below is a summary of the training results and key observations.

## Model Architecture

The GAT model consists of 6 graph attention layers with 8 attention heads. The architecture is as follows:

- **Layer 1**: GATConv(1433 input features, 64 output features) * 8 heads
- **Layer 2**: GATConv(64 * 8, 128) * 8 heads
- **Layer 3**: GATConv(128 * 8, 256) * 8 heads
- **Layer 4**: GATConv(256 * 8, 512) * 8 heads
- **Layer 5**: GATConv(512 * 8, 256) * 8 heads
- **Layer 6**: GATConv(256 * 8, num_classes)

The training uses the **CrossEntropyLoss** function with **Adam optimizer**. Dropout layers are applied to prevent overfitting.

## Conclusions
- **No Overfitting**: The model generalizes well as indicated by the validation loss and accuracy.
- **High Accuracy**: The final validation accuracy reaches **99.6%**, indicating strong performance on the node classification task.
- **Further Optimization**: The higher training loss compared to validation loss may suggest tuning parameters like the dropout rate or learning rate to optimize the training process.

  # Graph Attention Network (GAT) Community Detection Workflow

## Overview

This project demonstrates how to use a Graph Attention Network (GAT) for node classification and community detection. The workflow includes training a GAT model, extracting embeddings, constructing a similarity graph, and applying the Louvain algorithm to detect communities.

## Workflow

### 1. Train GAT Model

1. **Prepare the Dataset**:
   - Load and preprocess the dataset (e.g., CiteSeer, Cora).
   - Extract features and labels.

2. **Train the Model**:
   - Implement and train the Graph Attention Network (GAT) on the dataset.
   - Ensure the model is properly tuned for optimal performance.

3. **Extract Embeddings**:
   - After training, extract the second last layer embeddings from the GAT model. These embeddings capture rich node representations useful for community detection.

### 2. Construct Similarity Graph

1. **Compute Similarities**:
   - Use the extracted embeddings to compute pairwise similarities between nodes. This can be done using cosine similarity or another distance metric.

2. **Build the Graph**:
   - Construct a similarity graph where nodes are connected based on their computed similarities. Edges can be weighted according to similarity scores.

### 3. Apply Louvain Algorithm

1. **Run Louvain Algorithm**:
   - Apply the Louvain algorithm on the similarity graph to detect communities. This algorithm optimizes modularity to find clusters of nodes.

2. **Analyze Results**:
   - Evaluate the detected communities and analyze the modularity scores to assess the quality of the detected communities.


