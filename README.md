# ECE1508 Final Project - Coinsight
Coinsight is a deep learning based system for forecasting next-day cryptocurrency prices using only historical market data. The project addresses the inherent volatility and nonlinear dynamics of cryptocurrency markets by applying Recurrent Neural Networks (RNNs) sequence models to a sequence-to-one regression task. Given a fixed window of historical features, the model predicts the next-day closing price with a target relative error of 20% or better (currently we're using 10% which is twice as better as our initial proposal).

The system processes historical open, high, low, close prices, volume, returns, and related engineered features collected from the Bitcoin price dataset (sourced from Kaggle). After resampling, normalization, and feature preparation, the data is fed into two alternative architectures:

1. Baseline Model: **Elman RNN**

    A classical recurrent architecture that captures short-range temporal dependencies, serving as a benchmark for evaluating the effectiveness of our more deeper, enhanced sequence model.

2. Enhanced Model: **Stacked LSTM Network**

    A multi-layer LSTM with dropout between recurrent and fully connected layers. This architecture is designed to capture long-range temporal structure, stabilize training, and reduce overfitting which will be beneficial for financial time-series forecasting.

### Dependencies and Environment Setup
First, to run locally, you will need to clone the GitHub repository. This project requires Python 3.10+ and a small set of deep learning libraries (e.g., NumPy, Pandas, Scikit-learn, PyTorch, Matplotlib). To ensure reproducibility across systems, we provide a requirements.txt file. You may set up the environment using either pip. 

#### Steps:
#### 1. Clone the Repository with Git LFS Support

Since the dataset files are stored using Git Large File Storage (LFS), you need to install Git LFS before pulling the repo.

Install Git LFS

macOS (Homebrew):
```
brew install git-lfs
git lfs install
```

Ubuntu/Debian:
```
sudo apt update
sudo apt install git-lfs
git lfs install
```

Windows:

i. Download and run the Git LFS installer from https://git-lfs.github.com/

ii. Run:
```
git lfs install
```

Clone the repository and pull LFS files
```
git clone https://github.com/kris20012/crypto-price-prediction-rnn.git
cd crypto-price-prediction-rnn
git lfs pull
```

#### 2. Create and activate a virtual environment

a. For MacOS/Linux, first run:
```
python3 -m venv myenv
```

Then, run:
```
source myenv/bin/activate
```

b. For Windows/PowerShell, first run:
```
python -m venv myenv
```

Next, run:
```
myenv\Scripts\Activate
```

#### 3. Now, `cd` into the project repository and install the necessary dependencies. 
    
a. Run the following to update the pip package installer to latest version:
```
pip install --upgrade pip
```

b. Next, we can install the dependencies:
```
pip install -r requirements.txt
```

c. This installation can be verified using:
```
python -c "import torch, numpy, pandas; print('Environment is Correct & Ready!')"
```

#### 4. Finally, you can run each cell using an IDE like VSCode or launch the notebook using the following command:
```
jupyter notebook ECE1508_Final_Project.ipynb
```

