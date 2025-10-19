# kamma_2

This project aims to fit a deep neural network to predict the **probability of interaction (microscopic cross section)** as a function of the **energy of incident neutrons (MeV)** for **GOLD**.

The goal is to explore whether training a neural network can be faster and more efficient than performing multiple interpolations.

---

## 🧠 Project Overview

### `kamma_2.ipynb`
This notebook builds and trains a **Multi-Layer Perceptron (MLP)** neural network to approximate the interaction probability.  
- The model was trained three times on the training dataset to produce the log-log plot shown below.  
- For future training, it is recommended to **use all available data points** for training (no splitting).  
  The **validation set** should consist of the specific points you want to interpolate or compare with the trained model.

### `fourier.ipynb`
This notebook extends `kamma_2` by introducing **Fourier layers** to capture high-frequency behaviors in the data.  
This idea is based on the paper:  
📄 [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains (Tancik et al., 2020)](https://arxiv.org/pdf/2006.10739)  

The Fourier part still requires more tuning to properly capture all frequency components.

---

## ⚙️ Running the Code

You can run the code on **CPU** or **CUDA (GPU)** depending on your setup.

### CPU
Use the files in the `cpu` folder.  
Make sure all required libraries are installed.

### CUDA (Recommended)
For faster training and improved performance, it is recommended to run the network on **CUDA**.

### Install Dependencies
Run the following command to install the necessary Python packages:

```bash
pip install -r requirements.txt
