# Game of Life - CPU and CUDA Implementations

This repository contains two implementations of **Conway's Game of Life**:
- A **CPU-based version** that supports sequential processing, multithreading using `std::thread`, and parallelization using **OpenMP**.
- A **CUDA-based version** that leverages **GPU acceleration** using three memory types: **normal**, **pinned**, and **managed**.

---

## 📁 Repository Structure

├── basic_code/ # CPU-based implementation
│ └── main.cpp


├── cuda_code/ src # CUDA-based implementation
│               └── main.cpp


---

## 🧠 Features

### 🖥️ CPU Implementation (`basic_code/main.cpp`)
- Developed in C++ with support for:
  - **Sequential Processing** (`SEQ`)
  - **Multithreaded Processing** using `std::thread` (`THRD`)
  - **Parallel Processing** using OpenMP (`OMP`)
- Uses **SFML** for graphical visualization.
- Allows benchmarking over 100 generations.

### ⚙️ CUDA Implementation (`cuda_code/src/main.cpp`)
- Utilizes **NVIDIA CUDA** to accelerate computations.
- Supports:
  - **Normal device memory**
  - **Pinned host memory**
  - **Unified (Managed) memory**
- Renders output using **SFML**.
- Tracks performance over 100 iterations.

---

## 🧪 How to Build

### 🔧 Prerequisites
- C++ compiler with OpenMP support (`g++`, `clang++`, or MSVC)
- NVIDIA CUDA Toolkit (for `cuda_code`)
- [SFML](https://www.sfml-dev.org/download.php) library installed

### 💻 CPU Version (basic_code)

```bash
cd basic_code
g++ -std=c++17 -fopenmp main.cpp -o game_cpu -lsfml-graphics -lsfml-window -lsfml-system
./game_cpu -n 8 -t OMP -c 5 -x 800 -y 600

cd cuda_code
nvcc -std=c++14 main.cpp -o game_cuda -lsfml-graphics -lsfml-window -lsfml-system
./game_cuda -n 64 -t managed -c 5 -x 800 -y 600
