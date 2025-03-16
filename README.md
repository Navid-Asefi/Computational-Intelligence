# Genetic Algorithm: Selection Methods & Crossovers

## Overview
This project explores **Genetic Algorithms (GA)** and focuses on different **selection methods** and **crossover techniques** used in evolutionary computation. Genetic Algorithms are optimization heuristics inspired by the process of natural selection, commonly used for solving complex problems.

## Features
- Implementation of various **selection methods**:
  - Random Selection 
  - Roulette Wheel Selection
  - Rank-Based Selection
  - Tournament Selection
  - Truncation Selection
- Implementation of different **crossover techniques**:
  - Single-Point Crossover
  - Two-Point Crossover
  - Uniform Crossover
  - Arithmetic Crossover
- Flexible framework for experimentation and benchmarking
- Visualization of selection and crossover effects on population evolution

## Installation
Clone this repository and cd into it:
```sh
git clone https://github.com/Navid-Asefi/Computational-Intelligence
cd Computational-Intelligence
```

## Usage
Run the main script to experiment with different selection and crossover methods:
```sh
python main.py
```

## File Structure
```
📂 Computational-Intelligence
│-- 📜 main.py              # Main execution script
│-- 📜 selection.py         # Selection methods implementation
│-- 📜 crossover.py         # Crossover techniques implementation
│-- 📂 tools
│   ├── 📜 generation.py    # Generates the chromosomes needed
│   └── 📜 probability.py   # Computes the fitness and probabilities needed
│-- 📜 README.md            # Project documentation
```

## Contributions
Contributions are welcome! Feel free to fork the repo, make improvements, and submit a pull request.
