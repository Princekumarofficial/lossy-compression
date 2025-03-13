import numpy as np
import struct
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math

# ==== Float <-> Bit Conversion ====
def float_to_bits(f):
    [d] = struct.unpack(">Q", struct.pack(">d", f))
    return f'{d:064b}'

def bits_to_float(b):
    d = int(b, 2)
    return struct.unpack(">d", struct.pack(">Q", d))[0]

# ==== Compression ====
def compress_floats_bitpack(float_array, keep_bits=48, round_off=False):
    bitstream = ""
    for num in float_array:
        full_bits = float_to_bits(num)
        # Round off the last bit if needed
        if round_off:
            round_bit = full_bits[keep_bits]
            if round_bit == '1':
                rounded_bits = bin(int(full_bits[:keep_bits], 2) + 1)[2:].zfill(keep_bits)
                bitstream += rounded_bits
            else:
                bitstream += full_bits[:keep_bits]
        else:
            bitstream += full_bits[:keep_bits]

    padded_len = math.ceil(len(bitstream) / 8) * 8
    bitstream += '0' * (padded_len - len(bitstream))

    byte_array = bytearray()
    for i in range(0, len(bitstream), 8):
        byte = int(bitstream[i:i+8], 2)
        byte_array.append(byte)

    return byte_array

# ==== Decompression ====
def decompress_floats_bitpack(byte_array, num_floats, keep_bits=48):
    bitstream = ''.join(f'{byte:08b}' for byte in byte_array)
    recovered_floats = []
    for i in range(num_floats):
        start = i * keep_bits
        top_bits = bitstream[start:start+keep_bits]
        full_bits = top_bits.ljust(64, '0')
        recovered_floats.append(bits_to_float(full_bits))
    return recovered_floats

# ==== Save/Load ====
def save_bitpacked_file(byte_array, filename):
    with open(filename, 'wb') as f:
        f.write(byte_array)

def read_bitpacked_file(filename):
    with open(filename, 'rb') as f:
        return bytearray(f.read())

# ==== Analysis and Plotting ====
def analyze_distribution(name, data, recovered_np):
    PLOTS_DIR = "plots"

    mse = mean_squared_error(data, recovered_np)
    print(f"\n{name.capitalize()} Distribution Analysis:")
    print(f"Mean Squared Error (MSE): {mse:.6e}")

    # Histogram
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=50, alpha=0.6, label='Original', color='skyblue')
    plt.hist(recovered_np, bins=50, alpha=0.6, label='Compressed', color='salmon')
    plt.title(f"{name.capitalize()} Distribution: Histogram Comparison")
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{name}_histogram_comparison.png")
    plt.show()

    # Sorted Line Plot
    plt.figure(figsize=(10, 4))
    plt.plot(sorted(data), label='Original', linestyle='-')
    plt.plot(sorted(recovered_np), label='Compressed', linestyle='--')
    plt.title(f"{name.capitalize()} Sorted Value Plot")
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{name}_sorted_plot.png")
    plt.show()

    # Error Plot
    plt.figure(figsize=(10, 3))
    plt.plot(data - recovered_np, color='purple')
    plt.title(f"{name.capitalize()} Value Error (Original - Compressed)")
    plt.xlabel("Index")
    plt.ylabel("Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{name}_error_plot.png")
    plt.show()

# ==== Main Flow ====
if __name__ == "__main__":
    size = 1000
    keep_bits = 56 # Number of bits to keep - Make it smaller to increase compression
    # For high precision, keep_bits = 56
    # For limited storage, keep_bits = 48, or lower if you want more compression
    output_dir = "bin"

    uniform_data = np.random.uniform(1e10, 1e12, size)
    gaussian_data = np.random.normal(1e10, 1e12, size)
    exponential_data = np.random.exponential(1e10, size)

    all_data = {
        "uniform": uniform_data,
        "gaussian": gaussian_data,
        "exponential": exponential_data
    }

    for name, data in all_data.items():
        print(f"\n{name.capitalize()} Distribution:")
        data.astype(np.float64).tofile(f"{output_dir}/{name}_original.bin")

        compressed_bytes = compress_floats_bitpack(data, keep_bits=keep_bits, round_off=True)
        save_bitpacked_file(compressed_bytes, f"{output_dir}/{name}_compressed_bitpacked.bin")

        recovered = decompress_floats_bitpack(compressed_bytes, num_floats=size, keep_bits=keep_bits)
        recovered_np = np.array(recovered, dtype=np.float64)

        print("Original (first 5):", [f"{x:.10f}" for x in data[:5]])
        print("Recovered (first 5):", [f"{x:.10f}" for x in recovered_np[:5]])
        print(f"Original file size: {size * 8} bytes")
        print(f"Compressed file size: {len(compressed_bytes)} bytes")

        analyze_distribution(name, data, recovered_np)
