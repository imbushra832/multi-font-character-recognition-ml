

## 🧠  *Multi-Font Character Recognition: A Comparative Analysis of MLP, RBF, and SVM Networks*

### 📘 Project Overview

This project presents a comparative study of three machine learning architectures—**Multi-Layer Perceptron (MLP)**, **Radial Basis Function (RBF)** networks, and **Support Vector Machines (SVM)**—for the task of **multi-font character recognition**.
The models were evaluated on upper-case English characters rendered in **six different fonts**: *Courier, New York, Chicago, Geneva, Times,* and *Venice*.

The objective is to determine which architecture best generalizes across typographical variations and to analyze the relationship between model structure, hyperparameters, and recognition performance.

---

### 🎯 Objectives

* Develop and train three neural network models (MLP, RBF, and SVM) for character classification.
* Evaluate performance across six font types.
* Compare the impact of architectural and hyperparameter choices.
* Identify strengths and limitations of each model type.
* Explore ensemble modeling for performance improvement.

---

### 🧩 Dataset

* **Characters:** 26 uppercase English letters (A–Z)
* **Fonts:** Courier, New York, Chicago, Geneva, Times, Venice
* **Total Samples:** 156 (26 letters × 6 fonts)
* **Input Representation:** 14-dimensional feature vectors (from 18×18 normalized matrices)
* **Train/Test Split:** 50% / 50% (78 samples each)

Data preprocessing included:

* Feature extraction and dimensionality reduction
* Standardization with `StandardScaler`
* Randomized train-test splitting for evaluation consistency

---

### ⚙️ Model Architectures

#### 🧮 1. Multi-Layer Perceptron (MLP)

* Framework: `scikit-learn` (`MLPClassifier`)
* Explored configurations:

  * Hidden layers: (28,), (28, 14), (56,), (56, 28)
  * Activations: ReLU, tanh
  * Regularization (`alpha`): 0.0001–0.01
  * Learning rate schedules: constant, adaptive
* Best configuration:

  * 1 hidden layer (56 neurons), ReLU activation
  * Accuracy: **88.46%**

#### 🌐 2. Radial Basis Function (RBF) Network

* Custom RBF network implementation
* Tested configurations:

  * Random centers (26, 52, 78)
  * K-means and class-based initialization
  * Adaptive and per-class centers
* Key findings:

  * Random initialization (52 centers) achieved **88.46%**
  * Increasing centers improves performance until convergence (~2 per class)

#### ⚖️ 3. Support Vector Machine (SVM)

* Implemented with `scikit-learn`’s `SVC`
* Kernels tested: Linear, RBF, Polynomial
* Regularization (`C`): 0.1, 1, 10, 100
* Multi-class strategies: One-vs-One, One-vs-Rest
* Best configuration:

  * Linear kernel, C = 100
  * Accuracy: **87.18%**

---

### 📊 Results Summary

| Model                | Configuration                  | Accuracy   |
| :------------------- | :----------------------------- | :--------- |
| **MLP (optimal)**    | Single hidden layer (56, ReLU) | **0.8846** |
| **RBF (basic)**      | Random centers (52)            | **0.8846** |
| **SVM (linear)**     | C=100                          | 0.8718     |
| **RBF (k-means)**    | 52 centers                     | 0.8718     |
| **MLP (2-layer)**    | (28, 14)                       | 0.8590     |
| **SVM (RBF)**        | γ='scale'                      | 0.8333     |
| **RBF (26 centers)** | Random                         | 0.7564     |

**Highlights:**

* MLP and RBF achieved the highest accuracy (88.46%).
* SVM performed slightly below but remained highly competitive.
* Ensemble methods matched but did not exceed the best individual model performance.

---

### 🔍 Insights

* **Model capacity** (hidden neurons, number of centers) had a greater impact than hyperparameter tuning.
* **Linear separability** in the transformed feature space favored SVM’s linear kernel.
* **RBF networks** effectively captured non-linear font variations using distributed representation.
* **Feature representation** (14D) limits fine-grained visual distinctions (e.g., 'O' vs. 'Q').

---

### 🧩 Strengths and Limitations

| Model   | Strengths                                 | Limitations                                    |
| ------- | ----------------------------------------- | ---------------------------------------------- |
| **MLP** | Best overall accuracy; efficient training | Sensitive to hidden layer size                 |
| **RBF** | Interpretable centers; robust             | Requires tuning of centers and widths          |
| **SVM** | Simple; robust generalization             | Slightly lower accuracy; slower on multi-class |

---

### 💡 Future Work

1. **CNN-based architectures** — apply convolutional layers directly to character images.
2. **Feature engineering** — integrate geometric and statistical descriptors.
3. **Data augmentation** — increase robustness to noise and transformations.
4. **Explainability** — visualize neuron activations or RBF center responses.
5. **Larger datasets** — evaluate scalability across fonts and handwritten characters.

---

### 🧠 Key References

1. Lee, S. W. (1988). *Off-line recognition of totally unconstrained handwritten numerals using multilayer cluster neural network.* IEEE TPAMI.
2. Logar, A. M., Corwin, E. M., & Oldham, W. J. (1993). *A comparison of recurrent neural network learning algorithms.* IEEE ICNN.
3. Fujii, F., & Morita, M. (1984). *Feature extraction for handwritten character recognition.* ICPR.

---

### 📁 Repository Structure

```
multi-font-character-recognition/
│
├── data/
│   ├── characters/               # Raw and preprocessed samples
│   └── features.csv              # 14D extracted feature vectors
│
├── notebooks/
│   ├── mlp_model.ipynb
│   ├── rbf_model.ipynb
│   └── svm_model.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── rbf_network.py
│   ├── model_training.py
│   ├── evaluation_metrics.py
│   └── ensemble.py
│
├── results/
│   ├── confusion_matrices/
│   ├── model_performance.csv
│   └── training_logs/
│
├── README.md
└── requirements.txt
```

---

### ⚡ Requirements

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

---

### 🚀 Running the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/multi-font-character-recognition.git
   cd multi-font-character-recognition
   ```
2. Run the preprocessing:

   ```bash
   python src/data_preprocessing.py
   ```
3. Train the models:

   ```bash
   python src/model_training.py
   ```
4. View results:

   ```bash
   python src/evaluation_metrics.py
   ```

---



---

