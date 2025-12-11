# crypto-price-prediction-rnn
Deep learning project using RNNs to predict crypto prices

### Dependencies and Environment Setup
First, to run locally, you will need to clone the GitHub repository. This project requires Python 3.10+ and a small set of deep learning libraries (e.g., NumPy, Pandas, Scikit-learn, PyTorch, Matplotlib). To ensure reproducibility across systems, we provide a requirements.txt file. You may set up the environment using either pip. 

#### Steps:
1. From a terminal, `cd` into the directory where you would like to host the repository and run:
    ```
    git clone https://github.com/kris20012/crypto-price-prediction-rnn.git
    ```

2. Create and activate a virtual environment.

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

3. Now, `cd` into the project repository and install the necessary dependencies. 
    
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

4. Finally, you can run each cell using an IDE like VSCode or launch the notebook using the following command:
    ```
    jupyter notebook ECE1508_Final_Project.ipynb
    ```

