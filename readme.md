# Tweet Sentiment Analysis (SemEval Dataset/Word2Vec)

This repository contains the complete source code, datasets, and instructions to reproduce all experiments for tweet sentiment analysis using deep learning models on SemEval challenge datasets. This project is a direct implementation of the research paper titled **"Evaluation of Deep Learning Techniques in Sentiment Analysis from Twitter Data"**

---

## Table of Contents

- [Overview](#overview)
- [Package Contents](#package-contents)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Experiment Details](#experiment-details)
- [Results](#results)
- [Dependencies](#dependencies)
- [Citation & Credits](#citation--credits)

---

## Overview

This project implements and compares several deep learning models (CNN, LSTM, CNN+LSTM, 3-layer CNN+LSTM) for tweet sentiment classification (positive, neutral, negative) using data from multiple SemEval challenges. Both regional (region-based input) and non-regional (sentence-based input) variants are explored.

---

## Package Contents

- `dataset/` : All datasets used in the experiments
  - `tweets.txt`
  - `train_df.csv`
  - `test_df.csv`
- `requirements.txt` : List of required Python packages
- `Tweet-Sem_Eval.ipynb` : Jupyter notebook with source code
- `README.md` : This file

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SujalRatnaTamrakar/Tweet-Semantic-Eval.git
   cd Tweet-Semantic-Eval
   ```

2. **Set up a virtual environment (recommended)**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Data Preparation

1. Ensure the SemEval dataset files in the `dataset/` directory. Expected files:
   - `tweets.txt`
   - `train_df.csv`, `test_df.csv` 

No additional downloads are required.

---

### Jupyter Notebook

- Open `Tweet-Sem_Eval.ipynb` in Jupyter Notebook or Lab.
- Execute all cells sequentially.
- Output (training curves, confusion matrices, metrics) will display inline.

---

## Experiment Details

The repository contains implementations for:

- **CNN (Non-Regional / Regional)**
- **LSTM (Non-Regional / Regional)**
- **CNN+LSTM Hybrid (Non-Regional / Regional)**
- **3-layer CNN+LSTM**
- **Multi CNN + LSTM/BiLSTM**

All experiments use **Word2Vec embeddings**.

---

## Results

| Model                               |   Recall |   Precision |   F1 |   Accuracy |
|:------------------------------------|---------:|------------:|-----:|-----------:|
| Model 1: CNN (N-R)                  |     0.49 |        0.54 | 0.49 |       0.56 |
| Model 1: CNN (R)                    |     0.47 |        0.56 | 0.47 |       0.54 |
| Model 2: LSTM (N-R)                 |     0.5  |        0.54 | 0.51 |       0.56 |
| Model 2: LSTM (R)                   |     0.44 |        0.49 | 0.43 |       0.52 |
| Model 3: CNN + LSTM (N-R)           |     0.49 |        0.55 | 0.48 |       0.56 |
| Model 3: CNN + LSTM (R)             |     0.49 |        0.52 | 0.49 |       0.55 |
| Model 4: 3-layer CNN + LSTM (N-R)   |     0.47 |        0.5  | 0.48 |       0.53 |
| Model 4: 3-layer CNN + LSTM (R)     |     0.47 |        0.51 | 0.48 |       0.53 |
| Model 5: Multi CNN + LSTM (N-R)     |     0.49 |        0.57 | 0.49 |       0.56 |
| Model 5: Multi CNN + LSTM (R)       |     0.52 |        0.53 | 0.52 |       0.55 |
| Model 6: 3-layer CNN + BiLSTM (N-R) |     0.45 |        0.46 | 0.46 |       0.5  |
| Model 6: 3-layer CNN + BiLSTM (R)   |     0.47 |        0.5  | 0.47 |       0.53 |
| Model 7: Multi CNN + BiLSTM (N-R)   |     0.49 |        0.49 | 0.49 |       0.52 |
| Model 7: Multi CNN + BiLSTM (R)     |     0.5  |        0.54 | 0.5  |       0.55 |
---

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- plotly
- nltk
- gensim
- keras
- tensorflow
- scikit-learn
- wordcloud
- contractions

Install via:
```bash
pip install -r requirements.txt
```

---

## Citation & Credits

- Datasets: SemEval Twitter Sentiment Analysis Challenges
- Preprocessing and model code adapted from open-source libraries (see inline comments for attributions). Any external code is not counted towards original workload, as per submission guidelines.

### Original Research Paper

This project is based on the following research paper:

> **"Twitter Sentiment Analysis System Based on Deep Learning"**  
> Md. Kamrul Hasan, Md. Saiful Islam, Md. Khaled Chowdhury, and Md. Saiful Islam  
> *2019 International Conference on Electrical, Computer and Communication Engineering (ECCE)*  
> [Read the paper on IEEE Xplore](https://ieeexplore.ieee.org/document/8876896)


---

## Code Author

- **Author:** Sujal Ratna Tamrakar
- **Email:** [stamrakar1@student.gsu.edu]  
