# [**Precision Recovery in Lossy-Compressed Floating Point Data for High Energy Physics**](https://hepsoftwarefoundation.org/gsoc/2025/proposal_ATLAS_lossy_compression.html)**.**

### **1\. Objective**

Exploring **lossy floating-point compression** by **manipulating the least significant bits of the mantissa**. The main goal is to achieve storage savings while analyzing the impact on numerical precision and data distributions.

---

### **2\. Approach**

#### **2.1 Data Generation**

I started by generating relevantly large number of floating-point numbers from three distributions

- **Uniform Distribution**
- **Gaussian (Normal) Distribution**
- **Exponential Distribution**

Each dataset contains **1000 double-precision (64-bit) float values**.

#### **2.2 Compression Algorithm**

A custom bit-level compression function was implemented with the following characteristics:

- **Only the most significant `keep_bits` of the mantissa are retained (between 48–56 bits)**.
- The least significant bits are either zeroed or rounded off \- we get less error on rounding off.
- The modified bitstream is **bit-packed and saved as a binary file**.

  #### **2.3 Decompression**

During decompression, the **compressed mantissa bits are padded with zeros** to restore them to 64-bit float format. Although exact values cannot be recovered, this method approximates the original data sufficiently for many use cases.

---

### **3\. Implementation (Python)**

The compression and analysis code was written in Python using:

- **NumPy** for numerical operations
- **Matplotlib** for visualization
- **scikit-learn's MSE function** for error analysis
- **struct module** for precise bit-level manipulation
- **numba module** for faster code execution

Files are saved in `bin/` directory and plots in `plots/`.

---

### **4\. Results**

### **Case 1: `keep_bits = 56`**

#### **➤ Without Rounding**

| Distribution | Original File Size | Compressed File Size | Mean Squared Error (MSE) |
| ------------ | ------------------ | -------------------- | ------------------------ |
| Uniform      | 8000 bytes         | 7000 bytes           | 1.7335 × 10⁻⁴            |
| Gaussian     | 8000 bytes         | 7000 bytes           | 5.8759 × 10⁻⁴            |
| Exponential  | 8000 bytes         | 7000 bytes           | 1.1303 × 10⁻⁷            |

#### **➤ With Rounding**

| Distribution | Original File Size | Compressed File Size | Mean Squared Error (MSE) |
| ------------ | ------------------ | -------------------- | ------------------------ |
| Uniform      | 8000 bytes         | 7000 bytes           | 4.3675 × 10⁻⁵            |
| Gaussian     | 8000 bytes         | 7000 bytes           | 1.3434 × 10⁻⁴            |
| Exponential  | 8000 bytes         | 7000 bytes           | 2.9859 × 10⁻⁸            |

#### ---

### **Case 2: `keep_bits = 46`**

#### **➤ Without Rounding**

| Distribution | Original File Size | Compressed File Size | Mean Squared Error (MSE) |
| ------------ | ------------------ | -------------------- | ------------------------ |
| Uniform      | 8000 bytes         | 6000 bytes           | 1.1514 × 10¹             |
| Gaussian     | 8000 bytes         | 6000 bytes           | 3.5224 × 10¹             |
| Exponential  | 8000 bytes         | 6000 bytes           | 6.4230 × 10⁻³            |

#### **➤ With Rounding**

| Distribution | Original File Size | Compressed File Size | Mean Squared Error (MSE) |
| ------------ | ------------------ | -------------------- | ------------------------ |
| Uniform      | 8000 bytes         | 6000 bytes           | 2.8763 × 10⁰             |
| Gaussian     | 8000 bytes         | 6000 bytes           | 9.8995 × 10⁰             |
| Exponential  | 8000 bytes         | 6000 bytes           | 1.7852 × 10⁻³            |

### ---

### **Observations**

- Reducing precision (lower `keep_bits`) gives more compression but at the cost of increased error (MSE).
- Rounding before truncation consistently improves MSE across all distributions.
- The Exponential distribution shows minimal degradation even at lower precision.
- Uniform and Gaussian distributions suffer higher MSE at `keep_bits=46`, especially without rounding.
- Compression ratio improves from 12.5% (`56 bits → 7000 bytes`) to 25% (`46 bits → 6000 bytes`).

  ###

#### **Plots Generated**

- **Histogram Comparison**
- **Sorted Value Plot**
- **Error Plot (Original \- Compressed)**

All plots are saved as PNGs in the `plots/` directory.

---

### **5\. Key Observations**

- **Compression Savings**: Effective for applications with limited storage constraints.
- **Precision Trade-off**: Error is acceptable in many domains like visualization or approximate computing.
- **Exponential Distribution** shows greater sensitivity to bit truncation due to the long tail.
- ## **Rounding off LSBs** reduces cumulative error slightly better than pure truncation.

  ### **6\. Optimal Compression Recommendations**

| Use Case                          | Recommended `keep_bits` | Notes                                 |
| --------------------------------- | ----------------------- | ------------------------------------- |
| High-precision simulations        | 56+ bits                | Low error, less saving                |
| Data logging & telemetry          | 48 bits                 | Balance between size & accuracy       |
| Visualization or AI preprocessing | 40–48 bits              | Acceptable error for high compression |

---

### **7\. Improvements done**

- Used rounding techniques to reduce error and round the binary numbers before removing the fraction bits, used concepts like Guard Bit and also IEEE methods to round binary numbers.
- Used Numba to make the execution faster.
- Add support for **multi-threaded compression** for large datasets.
- ## TODO \- Will also implement a robust decompressor

### **8\. Running Instructions**

#### **Install Requirements**

```bash
pip install -r requirements.txt
```

#### **Run the Script**

To execute the compression and analysis script, navigate to the directory containing `task1v1.1.py` and run:

```bash
python task1v1.1.py
```

## Ensure that the `bin/` and `plots/` directories are present in the same directory as the script to store the output files and plots.

### **9\. References**

- IEEE 754 Double-Precision Floating-Point Format
- Data Compression Techniques in Scientific Computing
- scikit-learn, NumPy, Matplotlib Documentation
