# AHP (Analytic Hierarchy Process) Implementation

## Overview

This Python script implements the Analytic Hierarchy Process (AHP), a multi-criteria decision-making method. The AHP is used to prioritize alternatives based on multiple criteria by breaking down a complex decision-making problem into a hierarchy of goals, criteria, and alternatives.

## Features

- **Matrix Input**: Allows users to input pairwise comparison matrices for criteria and alternatives.
- **Weight Calculation**: Supports two methods for calculating weights:
  - **Arithmetic Mean Method**: Normalizes the matrix by columns and calculates the arithmetic mean of each row.
  - **Geometric Mean Method**: Multiplies the elements of each row, takes the nth root, and normalizes the resulting vector.
- **Consistency Check**: Performs a consistency check using the Consistency Index (CI) and Random Index (RI) to ensure the pairwise comparison matrices are consistent.
- **Final Weight Calculation**: Computes the final weights for each alternative based on the criteria weights.
- **Optimal Solution**: Identifies the optimal alternative for each goal based on the final weights.
- **Result Output**: Saves the results, including weights, consistency ratios, final weights, and optimal solutions, to a text file.

## Requirements

- Python 3.x
- NumPy

## Installation

1. Clone the repository or download the script.
2. Ensure you have Python 3.x installed.
3. Install the required dependencies using pip:

   ```bash
   pip install numpy
   ```

## Usage

1. Run the script:

   ```bash
   python main.py
   ```
2. Follow the prompts to input the number of goals, criteria, and alternatives.
3. Input the pairwise comparison matrices for the criteria and alternatives.
4. The script will perform the following steps:

   - Calculate weights using the selected method.
   - Perform a consistency check.
   - Calculate the final weights.
   - Identify the optimal solution.
   - Output the results to a file named `AHP_results.txt`.

## Example

### Input

- **Number of Goals**: 1
- **Number of Criteria**: 3
- **Number of Alternatives**: 4

### Pairwise Comparison Matrices

For example, the user will input matrices like:

```
1 2 3
0.5 1 2
0.333 0.5 1
```

### Output

The output will be saved in `AHP_results.txt` and will include:

- **Weights**: The calculated weights for each criterion and alternative.
- **Consistency Ratios**: The consistency ratios for each matrix.
- **Final Weights**: The final weights for each alternative.
- **Optimal Solution**: The optimal alternative for each goal.

## Methods

### Arithmetic Mean Method

1. **Normalize by Columns**: Divide each element by the sum of its column.
2. **Calculate Arithmetic Mean**: Compute the mean of each row to get the weights.

### Geometric Mean Method

1. **Multiply Row Elements**: Multiply the elements of each row.
2. **Take nth Root**: Take the nth root of the product.
3. **Normalize**: Normalize the resulting vector to get the weights.

## Consistency Check

- **Consistency Index (CI)**: Calculated as `(λmax - n) / (n - 1)`.
- **Random Index (RI)**: A pre-defined value based on the matrix size.
- **Consistency Ratio (CR)**: Calculated as `CI / RI`.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The AHP method was developed by Thomas L. Saaty.
- This implementation uses NumPy for matrix operations.

---

For any questions or issues, please open an issue on GitHub.
