# Financial Fraud Detection using Graph Neural Networks

![GNN Fraud Detection](img_landscape.webp)

This project showcases a cutting-edge financial fraud detection system built using Graph Neural Networks (GNNs). By leveraging graph-structured data and advanced techniques like spectral subgraph sampling and task-specific weight sharing, the system achieves a significant boost in precision and overall performance. The solution is containerized and deployed as a robust API for practical usability.

This project builds upon the ideas and architecture proposed in [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) (RGCN) and improvement upon this architecture using techniques from [UniGAD: Unifying Multi-level Graph Anomaly Detection](https://arxiv.org/abs/2411.06427)


## Overview
Fraud detection is a critical challenge in the financial industry, where conventional models often struggle to capture the complex relationships present in transaction data. This project introduces a GNN-based solution, specifically designed to model such intricate patterns effectively.

## Key features and Innovation
- **Precision Boost:** Achieved a 8% increase in fraud detection precision compared to baseline models like logistic regression and MLP.
- **Advanced Techniques:** Improved the F1 score by 32% and Average Precision (AP) score by 17% using:
    - **Spectral Subgraph Sampling**
    - **Task-Specific Weight Sharing**

- **Containerized API Deployment:** Easy-to-use API for fraud prediction, ready for integration into production systems.

## Performance Highlights
Since the dataset used is heavily imbalanced, we have to choose between Precsion or Recall and since we do not want to hinder user experience by incorrectly classifying the legitimate transactions as fraud, focusing on Precision seems like the right choice.

Even though focus is on precision, using techniques mentioned above, the F1 score and AP are improved significanlty without hurting the Precision.
| Metric | Logistic Regression | Neural Network | RGCN | RGCN + Unigad |
| :--: | :--: | :--: | :--: | :--: |
| Precision | 0.7890 | 0.8255 | 0.8966 | **0.9167**
| F1 Score | 0.3510 | 0.3891 | 0.4194 | **0.6079**
| Average Precision | 0.5120| 0.5800 | 0.6119 | **0.7397**
| AUC-ROC | 0.7930 | 0.8355 | 0.9203 | **0.9586**

## Architecture
1.	**Graph Construction**
     - Nodes represent entities (e.g., accounts, transactions).
     - Edges capture relationships (e.g., shared accounts, transactional links).
2. Model Training
    - **Spectral Subgraph Sampling:** Focuses on local graph structures for computational efficiency.
    - **Task-Specific Weight Sharing:** Enhances the model’s ability to differentiate fraud patterns. 

For more architecture and implementation detail please see [here](https://lakshyasingh.tech/fraud-detection).

## Setup and Installation
1.	Clone the Repository
```shell
git clone https://github.com/LakshyaSingh354/Financial-Fraud-Detection.git 
cd Financial-Fraud-Detection
```

2. Build Docker image
```shell
docker build -t fraud-detection .  
```

3. Run Docker Container
```shell
docker run -p 8080:8080 fraud-detection  
```

4. Acess the API
The API is accessible at http://localhost:8080/predict

## Future Enhancements
- Integrate temporal graph neural networks for time-series analysis.
- Develop explainability features to make predictions more interpretable.
- Extend to multi-relational graphs for more complex fraud detection scenarios.

## Acknowledgements
- [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103)
- [UniGAD: Unifying Multi-level Graph Anomaly Detection](https://arxiv.org/abs/2411.06427)
- Dataset: [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
- **DGL** For handling graph data.
- **PyTorch Geometric** for simplifying GNN implementation.
