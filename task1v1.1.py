import numpy as np
import struct
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math
import os

class FloatBitConverter:
    """
    Utility class for converting between IEEE 754 double-precision floating-point numbers and their bit representations.
    """
    @staticmethod
    def float_to_bits(f):
        """
        Convert a floating-point number to its 64-bit binary representation.
        
        Args:
            f (float): The floating-point number to convert
            
        Returns:
            str: A string of 64 bits representing the float in IEEE 754 format
        """
        [d] = struct.unpack(">Q", struct.pack(">d", f))
        return f'{d:064b}'

    @staticmethod
    def bits_to_float(b):
        """
        Convert a 64-bit binary representation back to a floating-point number.
        
        Args:
            b (str): A string of 64 bits representing a float in IEEE 754 format
            
        Returns:
            float: The reconstructed floating-point number
        """
        d = int(b, 2)
        return struct.unpack(">d", struct.pack(">Q", d))[0]

class FloatCompressor:
    """
    Compresses floating-point numbers by preserving only a specified number of significant bits.
    """
    def __init__(self, keep_bits=48, round_off=False):
        """
        Initialize the compressor with compression parameters.
        
        Args:
            keep_bits (int): Number of most significant bits to keep from each float
            round_off (bool): Whether to round values when truncating bits
        """
        self.keep_bits = keep_bits
        self.round_off = round_off

    def compress(self, float_array):
        """
        Compress an array of floating-point numbers.
        
        Args:
            float_array (array-like): Array of floating-point numbers to compress
            
        Returns:
            bytearray: Compressed data as a byte array
        """
        bitstream = ""
        for num in float_array:
            full_bits = FloatBitConverter.float_to_bits(num)
            if self.round_off:
                round_bit = full_bits[self.keep_bits]
                if round_bit == '1':
                    rounded_bits = bin(int(full_bits[:self.keep_bits], 2) + 1)[2:].zfill(self.keep_bits)
                    bitstream += rounded_bits
                else:
                    bitstream += full_bits[:self.keep_bits]
            else:
                bitstream += full_bits[:self.keep_bits]

        padded_len = math.ceil(len(bitstream) / 8) * 8
        bitstream += '0' * (padded_len - len(bitstream))

        byte_array = bytearray()
        for i in range(0, len(bitstream), 8):
            byte = int(bitstream[i:i+8], 2)
            byte_array.append(byte)

        return byte_array

class FloatDecompressor:
    """
    Decompresses floating-point numbers that were compressed using FloatCompressor.
    """
    def __init__(self, keep_bits=48):
        """
        Initialize the decompressor with decompression parameters.
        
        Args:
            keep_bits (int): Number of bits preserved for each float during compression
        """
        self.keep_bits = keep_bits

    def decompress(self, byte_array, num_floats):
        """
        Decompress a byte array back into an array of floating-point numbers.
        
        Args:
            byte_array (bytearray): The compressed data
            num_floats (int): Number of floats to recover from the compressed data
            
        Returns:
            list: List of recovered floating-point numbers
        """
        bitstream = ''.join(f'{byte:08b}' for byte in byte_array)
        recovered_floats = []
        for i in range(num_floats):
            start = i * self.keep_bits
            top_bits = bitstream[start:start+self.keep_bits]
            full_bits = top_bits.ljust(64, '0')
            recovered_floats.append(FloatBitConverter.bits_to_float(full_bits))
        return recovered_floats

class FileHandler:
    """
    Utility class for saving and reading binary files.
    """
    @staticmethod
    def save_file(byte_array, filename):
        """
        Save a byte array to a binary file.
        
        Args:
            byte_array (bytearray): The data to save
            filename (str): Path to the output file
        """
        with open(filename, 'wb') as f:
            f.write(byte_array)

    @staticmethod
    def read_file(filename):
        """
        Read a binary file into a byte array.
        
        Args:
            filename (str): Path to the file to read
            
        Returns:
            bytearray: The file contents as a byte array
        """
        with open(filename, 'rb') as f:
            return bytearray(f.read())

class DataAnalyzer:
    """
    Analyzes and visualizes the results of the compression/decompression process.
    """
    def __init__(self, plots_dir="plots"):
        """
        Initialize the analyzer with output directory for plots.
        
        Args:
            plots_dir (str): Directory to save generated plots
        """
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)

    def analyze_distribution(self, name, data, recovered_np):
        """
        Analyze and visualize the difference between original and recovered data.
        
        Args:
            name (str): Name of the distribution for labeling plots
            data (numpy.ndarray): Original data array
            recovered_np (numpy.ndarray): Recovered data after compression/decompression
        """
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
        plt.savefig(f"{self.plots_dir}/{name}_histogram_comparison.png")
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
        plt.savefig(f"{self.plots_dir}/{name}_sorted_plot.png")
        plt.show()

        # Error Plot
        plt.figure(figsize=(10, 3))
        plt.plot(data - recovered_np, color='purple')
        plt.title(f"{name.capitalize()} Value Error (Original - Compressed)")
        plt.xlabel("Index")
        plt.ylabel("Error")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/{name}_error_plot.png")
        plt.show()

class CompressionPipeline:
    """
    End-to-end pipeline for testing float compression on various data distributions.
    """
    def __init__(self, size=1000, keep_bits=56, output_dir="bin"):
        """
        Initialize the compression pipeline.
        
        Args:
            size (int): Number of samples to generate for each distribution
            keep_bits (int): Number of bits to preserve during compression
            output_dir (str): Directory to save binary files
        """
        self.size = size
        self.keep_bits = keep_bits
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.compressor = FloatCompressor(keep_bits=keep_bits, round_off=True)
        self.decompressor = FloatDecompressor(keep_bits=keep_bits)
        self.analyzer = DataAnalyzer()

    def run(self):
        """
        Run the compression pipeline on uniform, gaussian, and exponential data distributions.
        
        This method:
        1. Generates sample data from different distributions
        2. Compresses the data using bit reduction
        3. Decompresses the data
        4. Analyzes and visualizes the compression results
        """
        uniform_data = np.random.uniform(1e10, 1e12, self.size)
        gaussian_data = np.random.normal(1e10, 1e12, self.size)
        exponential_data = np.random.exponential(1e10, self.size)

        all_data = {
            "uniform": uniform_data,
            "gaussian": gaussian_data,
            "exponential": exponential_data
        }

        for name, data in all_data.items():
            print(f"\n{name.capitalize()} Distribution:")
            data.astype(np.float64).tofile(f"{self.output_dir}/{name}_original.bin")

            compressed_bytes = self.compressor.compress(data)
            FileHandler.save_file(compressed_bytes, f"{self.output_dir}/{name}_compressed_bitpacked.bin")

            recovered = self.decompressor.decompress(compressed_bytes, num_floats=self.size)
            recovered_np = np.array(recovered, dtype=np.float64)

            print("Original (first 5):", [f"{x:.10f}" for x in data[:5]])
            print("Recovered (first 5):", [f"{x:.10f}" for x in recovered_np[:5]])
            print(f"Original file size: {self.size * 8} bytes")
            print(f"Compressed file size: {len(compressed_bytes)} bytes")

            self.analyzer.analyze_distribution(name, data, recovered_np)

if __name__ == "__main__":
    pipeline = CompressionPipeline()
    pipeline.run()