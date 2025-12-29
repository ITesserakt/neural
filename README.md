# Neural Network Library in Rust

A high-performance neural network library implemented in Rust with automatic differentiation capabilities. This project provides a flexible and efficient framework for building and training neural networks, featuring a custom automatic differentiation system and support for MNIST digit classification.

## Features

- **Automatic Differentiation**: Custom implementation of reverse-mode automatic differentiation using Wengert lists (computational graph)
- **Flexible Network Architecture**: Build neural networks with configurable layers, activation functions, and weight initialization strategies
- **MNIST Support**: Built-in support for loading and training on the MNIST handwritten digit dataset
- **High Performance**: Leverages optimized BLAS libraries (Intel MKL) for efficient matrix operations
- **Serialization**: Save and load trained network parameters using efficient binary serialization
- **Interactive Training**: Command-line interface with interactive prompts for testing and evaluation
- **Progress Tracking**: Real-time progress bars and logging during training
- **Python Integration**: Uses PyO3 for seamless integration with Python datasets library

## Installation

### Prerequisites

- **Rust**: Install Rust using [rustup](https://rustup.rs/)
- **Python**: Python 3 with the `datasets` library installed

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd neural
```

2. Install Python dependencies:
```bash
pip install datasets numpy pillow
```

3. For Nix users, you can use the provided `shell.nix`:
```bash
nix-shell
```

4. Build the project:
```bash
cargo build --release
```

## Usage

### Basic Training

Train a neural network on the MNIST dataset:

```bash
cargo run --release -- \
    --dataset-path /path/to/mnist/dataset \
    --epoches 20 \
    --batch-size 256 \
    --learning-rate 0.1
```

### Command-Line Options

- `--dataset-path <PATH>`: Path to the MNIST dataset (required)
- `--cache-path <PATH>`: Path for caching datasets (default: `.cache`)
- `--parameters-path <PATH>`: Path to save/load network parameters (default: `params.dat`)
- `-p, --load-parameters-from-cache`: Load pre-trained parameters instead of training
- `-e, --epoches <NUM>`: Number of training epochs (default: 20)
- `-b, --batch-size <SIZE>`: Batch size for training (default: 256)
- `-l, --learning-rate <RATE>`: Learning rate for gradient descent (default: 0.1)

### Interactive Commands

After training (or when loading parameters), the program enters an interactive mode where you can:

- `test <index>`: Test the network on a specific test sample
- `test all`: Evaluate the network on all test samples and show statistics
- `save`: Save the current network parameters to disk
- `quit`: Exit the program

### Example Session

```bash
$ cargo run --release -- --dataset-path ./mnist --epoches 10
[Training progress bars...]
> test 42
Expected target probabilities:  [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00]
Predicted target probabilities: [0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.89, 0.02, 0.01]
Loss: 0.123
> save
> quit
```

## Architecture

### Automatic Differentiation

The project includes a custom automatic differentiation (AD) system in the `auto-differentiation` crate:

- **Reverse Mode AD**: Implemented using Wengert lists (computational graphs) for efficient backpropagation
- **Forward Mode AD**: Implemented using trace-based differentiation

### Neural Network

The `Network` type uses a builder pattern for constructing networks:

```rust
let network = Network::new(28 * 28)  // Input size (784 for MNIST)
    .push_hidden_layer(32, sigmoid_fn())  // Hidden layer with 32 neurons
    .push_output_layer(10, linear_fn())   // Output layer with 10 classes
    .map_output(Softmax);                 // Apply softmax to output
```

### Activation Functions

Available activation functions:

- `sigmoid_fn()`: Sigmoid activation
- `relu_fn()`: Rectified Linear Unit
- `leaky_relu()`: Leaky ReLU
- `linear_fn()`: Linear (identity) activation
- `softplus()`: Softplus activation
- `elu(alpha)`: Exponential Linear Unit
- `gaussian_fn()`: Gaussian activation
- `silu()`: Sigmoid Linear Unit (SiLU/Swish)

### Weight Initialization

Supported initialization strategies:

- `He`: He initialization
- `Xavier`: Xavier/Glorot initialization
- `Standard`: Standard normal distribution

### Output Functions

- `Softmax`: Softmax normalization for multi-class classification
- `Linear`: No transformation (for regression)

## Implementation Details

### Training Process

1. **Forward Pass**: Compute predictions for a batch of inputs
2. **Loss Calculation**: Compute cross-entropy loss between predictions and targets
3. **Backward Pass**: Use automatic differentiation to compute gradients
4. **Parameter Update**: Apply gradients using gradient descent with the specified learning rate

### Dataset Handling

- Datasets are loaded via Python's `datasets` library
- First load attempts to use cached serialized data
- If cache is missing, loads from Python and caches for future use
- Supports efficient zero-copy operations when possible

### Serialization

Network parameters are serialized using `postcard`, a compact binary format:
- Efficient storage of weights and biases
- Fast loading and saving
- Version-independent format

## Development

### Running Tests

```bash
cargo test
```

### Building with Optimizations

The project includes a `heavy` profile for maximum optimization:

```bash
cargo build --profile heavy
```

This profile enables:
- Link-time optimization (LTO)
- Maximum optimization level
- Stripped symbols
- Abort on panic

### Logging

The project uses `tracing` for structured logging:
- Progress bars via `indicatif`
- CSV output files for loss and predictions
- Configurable log levels via `RUST_LOG` environment variable

## Dependencies

### Core Dependencies

- `ndarray`: N-dimensional arrays and linear algebra
- `ndarray-linalg`: Linear algebra operations (BLAS/LAPACK)
- `ndarray-rand`: Random array generation
- `num-traits`: Numeric traits
- `smallvec`: Stack-allocated small vectors

### Automatic Differentiation

- `object-pool`: Object pooling for efficient memory management

### Python Integration

- `pyo3`: Rust-Python bindings
- `numpy`: NumPy array support

### CLI and Utilities

- `clap`: Command-line argument parsing
- `serde`: Serialization framework
- `postcard`: Compact binary serialization
- `indicatif`: Progress bars
- `tracing`: Structured logging

## Performance Considerations

- Uses Intel MKL for optimized BLAS operations
- Efficient memory layout with `ndarray`
- Batch processing for better cache locality
- Object pooling in the AD system to reduce allocations
- SmallVec for stack-allocated small collections

## Limitations and Future Work

- Currently focused on fully-connected (dense) layers
- Single-threaded training (no parallel batch processing)
- Fixed batch size during training
- Limited to MNIST dataset structure (though extensible)

Potential enhancements:
- Convolutional layers
- Recurrent layers (RNN, LSTM, GRU)
- Multi-threaded training
- GPU acceleration
- More optimizers (Adam, RMSprop, etc.)
- Additional loss functions

## Acknowledgments

The automatic differentiation implementation is based on concepts from the `easy-ml` library and implements reverse-mode automatic differentiation using Wengert lists (computational graphs).

