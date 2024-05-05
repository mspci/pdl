# Program Setup Guide

## Setup Steps:

1. Open a conda terminal.

2. Navigate to the desired directory (e.g., desktop or any other location):

    ```bash
    cd Desktop
    ```

3. Clone the repository from GitHub:

    ```bash
    git clone https://github.com/mspci/pdl
    cd pdl
    ```

4. Create a conda environment named "maskrcnn" with Python version 3.7.11:

    ```bash
    conda create -n maskrcnn python=3.7.11 -y
    ```

5. Activate the newly created conda environment:

    ```bash
    conda activate maskrcnn
    ```

6. Install the required Python packages listed in `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

7. Install protobuf using either conda or pip:

    ```bash
    conda install protobuf
    ```
   
   or
   
    ```bash
    pip install protobuf
    ```

## Training the Program:

To perform training, navigate to the repository directory 'pdl' in the terminal and run the following command:

```bash
python remote_sensing_training.py
