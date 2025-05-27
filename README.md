# HADS

## Introduction
This study introduces a prediction model driven by Hyperedge-topology-enhanced Hypergraph Learning and Adaptive Multi-graph Transformer Learning (HADS) for drug-side effect association prediction.

---

## Catalogs
- /data:  Contains the datasets used in the model.
- /code: Contains the implementation code for HADS.
- data_process.py: Processes drug and side effect similarities, associations, features, and adjacency matrices (hypergraph and line graph) for training and testing.
- HGCN.py,LGCN.py,AMT.py，FLGN: Implementations of hypergraph convolution, line graph convolution, adaptive multi-graph Transformer, and feature-level gating network, respectively.
- model_all.py: Full model integration.
- main.py: Parameter settings, model training and testing.

---

## Dataset
- drug_drug_sim_dis.txt: Drug functional similarity
- mat_drug_se.txt: Drug-side effect associations
- se_seSmilirity.csv: Side effect similarity
- Simularity_Matrix_Drugs.txt: Drug structural similarity

---

## Environment  
The HADS code has been implemented and tested in the following development environment: 

- Python == 3.8.10 
- Matplotlib == 3.5.2
- PyTorch == 2.0.0 
- NumPy == 1.22.4

---

## How to Run the Code
1. **Data preprocessing: Constructs the adjacency matrices, embeddings, and other inputs for training the model**.  
    ```bash
    python data_process.py
    ```  

2. **Train and test the model**.  
    ```bash
    python main.py
    ```  


