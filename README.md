# Parallel Neural Network Training using MPI

  University of Bayreuth, 2025 -2026
**Author:** Rawen Jendoubi  

A from-scratch implementation of a feedforward neural network trained via backpropagation on the MNIST digit recognition dataset, with a sequential baseline and a parallelized version using MPI for multi-process CPU training.

---

## Project Structure

```
├── NN_seq.c                        # Sequential neural network (baseline)
├── NN_mpi_2.c                      # Parallel neural network using MPI
├── param_parser.h                  # Struct definitions and function declarations
├── param_parser_fseek_sscanf.c     # Config file parser and dataset loader
│
├── mnist_bench_case1.cfg           # Config: MNIST benchmark (784→512→128→10)
├── brain_bench_case1.cfg           # Config: Brain tumor dataset (784→512→128→4)
│
├── mnist_train_dataset.dat         # Training samples (1000 samples)
├── mnist_train_labels.dat          # Training labels (one-hot encoded)
├── mnist_train_pixels.dat          # Training pixel values (normalized)
├── mnist_test_dataset.dat          # Test samples (1000 samples)
├── mnist_test_labels.dat           # Test labels
├── mnist_test_pixels.dat           # Test pixel values
│
├── mnist_pre_processing_v1.ipynb   # Full dataset conversion (CSV → .dat)
├── mnist_conversion_train.ipynb    # Train split conversion notebook
└── mnist_conversion_test.ipynb     # Test split conversion notebook
```

---

## Network Architecture

| Layer         | Neurons |
|---------------|---------|
| Input         | 784 (28×28 pixels) |
| Hidden Layer 1| 512     |
| Hidden Layer 2| 128     |
| Output        | 10 (digits 0–9) |

- **Activation function:** Sigmoid
- **Weight initialization:** Uniform random in [-0.5, 0.5]
- **Training:** Backpropagation with gradient descent

---

## Requirements

- GCC (C99 or later)
- OpenMPI or MPICH
- Python 3.x + NumPy (for dataset preprocessing only)

Install MPI on Ubuntu/Debian:

```bash
sudo apt install openmpi-bin libopenmpi-dev
```

---

## Dataset Preprocessing

The raw MNIST CSV files must be converted to `.dat` format before training.  
Run the Jupyter notebooks in order:

```bash
jupyter notebook mnist_pre_processing_v1.ipynb       # Full dataset
jupyter notebook mnist_conversion_train.ipynb         # Training split
jupyter notebook mnist_conversion_test.ipynb          # Test split
```

This produces the `.dat` files with one-hot labels and normalized pixel values (0–1).

---

## Configuration

Training parameters are defined in `.cfg` files:

```
# input and output neurons
784 10
# hidden layers
512 128
# training parameters (learning rate, epochs)
0.1 100
# Training & Testing Samples
20 10
# data file
mnist_dataset.dat
```

---

## Compilation

**Sequential version:**
```bash
gcc -O2 -o nn_seq NN_seq.c param_parser_fseek_sscanf.c -lm
```

**MPI parallel version:**
```bash
mpicc -O2 -o nn_mpi NN_mpi_2.c param_parser_fseek_sscanf.c -lm
```

---

## Usage

**Run sequential:**
```bash
./nn_seq
```

**Run parallel with N processes:**
```bash
mpirun -n 4 ./nn_mpi
```

---

## How the Parallelization Works

The MPI version distributes the training workload across processes:

1. **Process 0** loads the config file and broadcasts parameters to all processes via `MPI_Bcast`
2. Each process handles a **subset of training samples** (data parallelism)
3. After each epoch, gradients are **reduced across all processes** using `MPI_Allreduce` to synchronize weight updates
4. All processes maintain identical network weights after each synchronization step

---

## Output Example

```
Running with 4 MPI process(es)

=== Config loaded successfully ===
  input_neurons    = 784
  hidden_neurons_1 = 512
  hidden_neurons_2 = 128
  output_neurons   = 10
  learning_rate    = 0.1000
  epochs           = 100

Epoch 0,  Error: 0.412300, Time: 0.0241 seconds
Epoch 10, Error: 0.198500, Time: 0.0198 seconds
...

=== TRAINING COMPLETED ===
Total training time: 2.1043 seconds

Sample 0
Actual: 7, Predicted: 7, Result: CORRECT
```

---

## Acknowledgements

I would like to express my sincere gratitude to **Prof. Matthias  Korch** for his guidance,
support and for sharing his valuable knowledge, expertise and insights throughout
the development of this work. I would also like to thank **Mr. Werner** for his
assistance and contributions during this project.

---

## License

For academic use only — University of Bayreuth, 2025-2026.
