# 🧠 NLP HOMEWORK 4 – RNNs, Transformers & Attention (Part II)

### 📘 Student Information
**Name:** Venkata Nanda Krishna Yaram  
**Course:** Natural Language Processing  
**ST,ID:** 700765514

---

## 🚀 Overview
This repository implements three core deep-learning components used in Natural Language Processing:

1. **Character-Level RNN Language Model** – Builds a simple character-based LSTM that predicts the next character.
2. **Mini Transformer Encoder** – Demonstrates how self-attention and multi-head mechanisms create contextual embeddings.
3. **Scaled Dot-Product Attention** – Implements the mathematical attention operation with numerical stability analysis.

All codes are written in **PyTorch** and tested on **Google Colab**.

---

## 🧩 Question 1 – Character-Level RNN Language Model
### 🎯 Goal
Train a small RNN that predicts the next character given previous characters.

### 🔧 Model Architecture
### ⚙️ Training Setup
- **Hidden size:** 128  
- **Sequence length:** 80  
- **Batch size:** 64  
- **Optimizer:** Adam  
- **Loss:** Cross-Entropy  
- **Teacher forcing:** Enabled  
- **Epochs:** 10  
- **Temperature sampling:** τ = 0.7 / 1.0 / 1.2  
- **Dataset:** Toy corpus (“hello help helmet…”) or any small ~100 KB text file  

### 📈 Results
- Plots **training vs validation loss curves** using Matplotlib.  
- Generates **text samples** at different temperature values.  

### 💡 Reflection
- Longer sequence length = better context learning but slower training.  
- Larger hidden size = smoother, more coherent text but risk of overfitting.  
- Lower temperature (τ = 0.7) → predictable and repetitive.  
- Higher temperature (τ = 1.2) → creative and diverse but less accurate.  
- Teacher forcing speeds early learning and stabilizes training.

---

## ⚡ Question 2 – Mini Transformer Encoder
### 🎯 Goal
Build and visualize a compact Transformer Encoder that processes batches of short sentences.

### 🔧 Components
- **Token Embeddings**  
- **Sinusoidal Positional Encoding**  
- **Multi-Head Self-Attention** (4 heads)  
- **Feed-Forward Network (FFN)**  
- **Add & LayerNorm**  

### 🧩 Implementation Flow
1. Tokenize 10 short sentences.  
2. Apply embeddings + positional encoding.  
3. Compute multi-head self-attention.  
4. Pass through FFN + Add & Norm.  
5. Visualize attention weights as a heatmap.  

### 📊 Outputs
- Input tokens and their final contextual embeddings.  
- Attention heatmap showing how each word attends to others in the sentence.  

### 🧠 Insights
- Multi-head attention captures different relations (e.g., syntax, coreference).  
- Add & Norm ensures gradient stability and faster convergence.  
- Positional encoding preserves token order for non-sequential attention.

---

## 🎯 Question 3 – Scaled Dot-Product Attention
### 🎯 Goal
Implement the mathematical definition of attention from “Attention is All You Need”.

### 🧮 Formula
\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V
\]

### ⚙️ Implementation Steps
1. Generate random Q, K, V tensors (shape = (batch, seq_len, d_k)).  
2. Compute raw attention scores and scaled scores.  
3. Apply softmax to obtain attention weights.  
4. Multiply weights × V to get context vectors.  
5. Compare stability before and after scaling.  

### 📈 Outputs
- Unscaled vs scaled attention scores  
- Attention weight matrix (softmax output)  
- Final output vectors  
- Numeric range comparison showing why division by √dₖ stabilizes training  

### 🧠 Observation
Without scaling, large dot-products cause the softmax to saturate.  
Scaling by √dₖ keeps values within a stable range and preserves gradient flow.

---

