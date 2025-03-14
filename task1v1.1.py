import numpy as np
import struct
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from numba import njit
import os
import gzip
import zlib
import lzma
import bz2

@njit
def float_to_bits_numba(f):
    return np.uint64(np.asarray(f, dtype=np.float64).view(np.uint64))

@njit
def compress_floats_numba(float_array, keep_bits, round_off):
    n = len(float_array)
    compressed_ints = np.empty(n, dtype=np.uint64)

    for i in range(n):
        bits = float_to_bits_numba(float_array[i])

        if round_off:
            guard_pos = 64 - keep_bits - 1
            guard_bit = (bits >> guard_pos) & 1 if guard_pos >= 0 else 0
            sticky_bits = bits & ((1 << guard_pos) - 1) if guard_pos > 0 else 0
            sticky_bit = 1 if sticky_bits != 0 else 0

            preserved_bits = bits >> (64 - keep_bits)
            lsb = preserved_bits & 1

            if guard_bit == 1 and (sticky_bit == 1 or lsb == 1):
                preserved_bits += 1
                if preserved_bits >= (1 << keep_bits):  # overflow
                    preserved_bits -= 1
        else:
            preserved_bits = bits >> (64 - keep_bits)

        compressed_ints[i] = preserved_bits

    # Bitpack into a uint8 buffer
    total_bits = 0
    bitstream = 0
    max_bytes = ((keep_bits * n + 7) // 8)  # max size in bytes
    output_bytes = np.zeros(max_bytes, dtype=np.uint8)
    byte_idx = 0

    for i in range(n):
        bitstream = (bitstream << keep_bits) | compressed_ints[i]
        total_bits += keep_bits

        while total_bits >= 8:
            total_bits -= 8
            output_bytes[byte_idx] = (bitstream >> total_bits) & 0xFF
            byte_idx += 1

    # Handle leftover bits
    if total_bits > 0:
        output_bytes[byte_idx] = (bitstream << (8 - total_bits)) & 0xFF
        byte_idx += 1

    return output_bytes[:byte_idx]

class FloatBitConverter:
    """
    Utility class for converting between IEEE 754 double-precision floating-point numbers and their bit representations.
    """
    @staticmethod
    def float_to_bits(f):
        """Convert float64 to uint64 bit representation."""
        return struct.unpack(">Q", struct.pack(">d", f))[0]

    @staticmethod
    def bits_to_float(b):
        """Convert uint64 bit representation back to float64."""
        return struct.unpack(">d", struct.pack(">Q", b))[0]

class FloatCompressor:
    """
    Compresses floating-point numbers by preserving only a specified number of significant bits,
    with optional IEEE 754-aware rounding using guard and sticky bits.
    """
    def __init__(self, keep_bits=48, round_off=True):
        """
        Initialize the compressor with compression parameters.

        Args:
            keep_bits (int): Number of most significant bits to keep from each float.
            round_off (bool): Whether to use guard + sticky bit rounding when truncating.
        """
        self.keep_bits = keep_bits
        self.round_off = round_off
        self.bit_mask = (1 << keep_bits) - 1

    def compress(self, float_array):
        compressed_np_bytes = compress_floats_numba(float_array, self.keep_bits, self.round_off)
        return bytearray(compressed_np_bytes)

class FloatDecompressor:
    """
    Decompresses floating-point numbers that were compressed using FloatCompressor.
    """
    def __init__(self, keep_bits=48):
        """
        Initialize the decompressor with decompression parameters.

        Args:
            keep_bits (int): Number of bits preserved for each float during compression.
        """
        self.keep_bits = keep_bits

    def decompress(self, byte_array, num_floats):
        bitstream = 0
        total_bits = 0
        byte_idx = 0
        recovered_floats = []

        for _ in range(num_floats):
            while total_bits < self.keep_bits and byte_idx < len(byte_array):
                bitstream = (bitstream << 8) | byte_array[byte_idx]
                total_bits += 8
                byte_idx += 1

            total_bits -= self.keep_bits
            top_bits = (bitstream >> total_bits) & ((1 << self.keep_bits) - 1)
            bitstream &= (1 << total_bits) - 1  # remove used bits

            full_bits = top_bits << (64 - self.keep_bits)
            recovered_floats.append(FloatBitConverter.bits_to_float(full_bits))

        return recovered_floats

class EnhancedCompressor(FloatCompressor):
    """
    Enhanced compressor that applies additional compression after bit-packing.
    """
    def __init__(self, keep_bits=48, round_off=True, algorithm='gzip', compression_level=9):
        """
        Initialize the enhanced compressor.
        
        Args:
            keep_bits (int): Number of bits to preserve from each float
            round_off (bool): Whether to use guard + sticky bit rounding
            algorithm (str): Secondary compression algorithm ('gzip', 'zlib', 'lzma', 'bz2')
            compression_level (int): Compression level (1-9, with 9 being highest)
        """
        super().__init__(keep_bits, round_off)
        self.algorithm = algorithm
        self.compression_level = compression_level
    
    def compress(self, float_array):
        # First, apply bit-packing compression
        bit_packed = super().compress(float_array)
        
        # Then apply secondary compression
        if self.algorithm == 'gzip':
            return gzip.compress(bit_packed, compresslevel=self.compression_level)
        elif self.algorithm == 'zlib':
            return zlib.compress(bit_packed, level=self.compression_level)
        elif self.algorithm == 'lzma':
            return lzma.compress(bit_packed, preset=self.compression_level)
        elif self.algorithm == 'bz2':
            return bz2.compress(bit_packed, compresslevel=self.compression_level)
        else:
            return bit_packed  # Default: return just the bit-packed data

class EnhancedDecompressor(FloatDecompressor):
    """
    Enhanced decompressor that handles additional compression layers.
    """
    def __init__(self, keep_bits=48, algorithm='gzip'):
        """
        Initialize the enhanced decompressor.
        
        Args:
            keep_bits (int): Number of bits preserved during compression
            algorithm (str): Secondary compression algorithm used ('gzip', 'zlib', 'lzma', 'bz2')
        """
        super().__init__(keep_bits)
        self.algorithm = algorithm
    
    def decompress(self, byte_array, num_floats):
        # First, undo the secondary compression
        if self.algorithm == 'gzip':
            bit_packed = gzip.decompress(byte_array)
        elif self.algorithm == 'zlib':
            bit_packed = zlib.decompress(byte_array)
        elif self.algorithm == 'lzma':
            bit_packed = lzma.decompress(byte_array)
        elif self.algorithm == 'bz2':
            bit_packed = bz2.decompress(byte_array)
        else:
            bit_packed = byte_array  # If no secondary compression was used
        
        # Then, apply the original decompression
        return super().decompress(bit_packed, num_floats)

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
    def __init__(self, plots_dir="plots", results_dir="results"):
        """
        Initialize the analyzer with output directory for plots and results.
        
        Args:
            plots_dir (str): Directory to save generated plots
            results_dir (str): Directory to save results markdown files
        """
        self.plots_dir = plots_dir
        self.results_dir = results_dir
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        self.results_data = {}

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
        plt.hist([data, recovered_np], bins=50, label=['Original', 'Compressed'], color=['skyblue', 'salmon'], alpha=0.6)
        plt.title(f"{name.capitalize()} Distribution: Histogram Comparison")
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/{name}_histogram_comparison.png")
        plt.close()

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
        plt.close()

        # Error Percent Plot
        error_percent = 100 * (data - recovered_np) / data
        plt.figure(figsize=(10, 3))
        plt.plot(error_percent, color='purple')
        plt.title(f"{name.capitalize()} Value Error Percent (Original - Compressed) / Original * 100")
        plt.xlabel("Index")
        plt.ylabel("Error Percent")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/{name}_error_percent_plot.png")
        plt.close()
        
        return mse

    def add_result(self, name, algorithm, keep_bits, round_off, original_size, compressed_size, mse):
        """
        Add a compression result to the results data.
        
        Args:
            name (str): Distribution name
            algorithm (str): Compression algorithm used
            keep_bits (int): Number of bits preserved
            round_off (bool): Whether rounding was used
            original_size (int): Original data size in bytes
            compressed_size (int): Compressed data size in bytes
            mse (float): Mean squared error
        """
        key = f"keep_bits={keep_bits}_round_off={round_off}"
        if key not in self.results_data:
            self.results_data[key] = {}
        
        if algorithm not in self.results_data[key]:
            self.results_data[key][algorithm] = {}
            
        self.results_data[key][algorithm][name] = {
            "original_size": original_size,
            "compressed_size": compressed_size,
            "mse": mse,
            "compression_percent": compressed_size / original_size * 100,
            "R": original_size / compressed_size
        }

    def generate_markdown(self, filename="compression_results.md"):
        """
        Generate a markdown file with all the results.
        
        Args:
            filename (str): Name of the markdown file to generate
        """
        with open(f"{self.results_dir}/{filename}", "w") as f:
            f.write("# Float Compression Results\n\n")
            
            for config, algorithms in self.results_data.items():
                f.write(f"## Configuration: {config}\n\n")
                
                for algorithm, distributions in algorithms.items():
                    f.write(f"### Algorithm: {algorithm}\n\n")
                    
                    # Create table header
                    f.write("| Distribution | Original Size | Compressed Size | Compression % | R | MSE |\n")
                    f.write("| ------------ | ------------- | --------------- | ------------- | --- | --- |\n")
                    
                    # Add rows for each distribution
                    for name, results in distributions.items():
                        f.write(f"| {name.capitalize()} | {results['original_size']} bytes | {results['compressed_size']} bytes | ")
                        f.write(f"{results['compression_percent']:.2f}% | {results['R']:.2f} | {results['mse']:.6e} |\n")
                    
                    f.write("\n")
                
                # Add section for plots
                f.write("#### Plots\n\n")
                for name in distributions.keys():
                    f.write(f"##### {name.capitalize()} Distribution\n\n")
                    f.write(f"![{name} Histogram](../{self.plots_dir}/{name}_histogram_comparison.png)\n\n")
                    f.write(f"![{name} Sorted Plot](../{self.plots_dir}/{name}_sorted_plot.png)\n\n")
                    f.write(f"![{name} Error Plot](../{self.plots_dir}/{name}_error_percent_plot.png)\n\n")
            
            print(f"Markdown file generated: {self.results_dir}/{filename}")

class CompressionPipeline:
    """
    End-to-end pipeline for testing float compression on various data distributions.
    """
    def __init__(self, size=1000, keep_bits=56, output_dir="bin", round_off=True):
        """
        Initialize the compression pipeline.
        
        Args:
            size (int): Number of samples to generate for each distribution
            keep_bits (int): Number of bits to preserve during compression
            output_dir (str): Directory to save binary files
            round_off (bool): Whether to use rounding during compression
        """
        self.size = size
        self.keep_bits = keep_bits
        self.round_off = round_off
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.compressor = FloatCompressor(keep_bits=keep_bits, round_off=round_off)
        self.decompressor = FloatDecompressor(keep_bits=keep_bits)
        self.analyzer = DataAnalyzer()

    def run(self, all_data):
        """
        Run the compression pipeline on provided data distributions.
        
        This method:
        1. Compresses the data using bit reduction
        2. Decompresses the data
        3. Analyzes and visualizes the compression results
        """
        original_size = self.size * 8  # 8 bytes per float64

        for name, data in all_data.items():
            print(f"\n{name.capitalize()} Distribution:")
            data_file = f"{self.output_dir}/{name}_original.bin"
            data.astype(np.float64).tofile(data_file)

            compressed_bytes = self.compressor.compress(data)
            compressed_file = f"{self.output_dir}/{name}_compressed_bitpacked.bin"
            FileHandler.save_file(compressed_bytes, compressed_file)
            compressed_size = len(compressed_bytes)

            recovered = self.decompressor.decompress(compressed_bytes, num_floats=self.size)
            recovered_np = np.array(recovered, dtype=np.float64)

            print("Original (first 5):", [f"{x:.10f}" for x in data[:5]])
            print("Recovered (first 5):", [f"{x:.10f}" for x in recovered_np[:5]])
            print(f"Original file size: {original_size} bytes")
            print(f"Compressed file size: {compressed_size} bytes")
            print(f"Compression ratio: {original_size / compressed_size:.2f}")
            print(f"Compression percentage: {compressed_size / original_size * 100:.2f}%")

            mse = self.analyzer.analyze_distribution(name, data, recovered_np)
            self.analyzer.add_result(
                name=name,
                algorithm="bitpacking",
                keep_bits=self.keep_bits,
                round_off=self.round_off,
                original_size=original_size,
                compressed_size=compressed_size,
                mse=mse
            )
        
        # Generate markdown file with results
        self.analyzer.generate_markdown(f"compression_results_keep_bits_{self.keep_bits}_round_{self.round_off}.md")
    
    def run_enhanced(self, all_data):
        """
        Run the enhanced compression pipeline with secondary compression.
        """
        # Test different compression algorithms
        algorithms = ['gzip', 'zlib', 'lzma', 'bz2']
        original_size = self.size * 8  # 8 bytes per float64
        
        for name, data in all_data.items():
            print(f"\n{name.capitalize()} Distribution:")
            data_file = f"{self.output_dir}/{name}_original.bin"
            data.astype(np.float64).tofile(data_file)
            
            # Original bit-packing only
            compressor = FloatCompressor(keep_bits=self.keep_bits, round_off=self.round_off)
            decompressor = FloatDecompressor(keep_bits=self.keep_bits)
            
            bit_packed = compressor.compress(data)
            FileHandler.save_file(bit_packed, f"{self.output_dir}/{name}_bitpacked.bin")
            bit_packed_size = len(bit_packed)
            
            print(f"Original size: {original_size} bytes")
            print(f"Bit-packed size: {bit_packed_size} bytes ({bit_packed_size/original_size*100:.2f}%)")

            recovered = decompressor.decompress(bit_packed, num_floats=self.size)
            recovered_np = np.array(recovered, dtype=np.float64)
            mse = self.analyzer.analyze_distribution(name, data, recovered_np)
            
            self.analyzer.add_result(
                name=name,
                algorithm="bitpacking",
                keep_bits=self.keep_bits,
                round_off=self.round_off,
                original_size=original_size,
                compressed_size=bit_packed_size,
                mse=mse
            )
            
            # Test each secondary compression algorithm
            for algo in algorithms:
                enhanced_compressor = EnhancedCompressor(
                    keep_bits=self.keep_bits, 
                    round_off=self.round_off,
                    algorithm=algo
                )
                enhanced_decompressor = EnhancedDecompressor(
                    keep_bits=self.keep_bits,
                    algorithm=algo
                )
                
                compressed = enhanced_compressor.compress(data)
                FileHandler.save_file(compressed, f"{self.output_dir}/{name}_{algo}.bin")
                compressed_size = len(compressed)
                
                recovered = enhanced_decompressor.decompress(compressed, num_floats=self.size)
                recovered_np = np.array(recovered, dtype=np.float64)
                
                print(f"{algo} size: {compressed_size} bytes ({compressed_size/original_size*100:.2f}%)")
                
                # Verify accuracy is maintained
                mse = mean_squared_error(data, recovered_np)
                print(f"{algo} MSE: {mse:.6e}")
                
                self.analyzer.add_result(
                    name=name,
                    algorithm=algo,
                    keep_bits=self.keep_bits,
                    round_off=self.round_off,
                    original_size=original_size,
                    compressed_size=compressed_size,
                    mse=mse
                )
        
        # Generate markdown file with results
        self.analyzer.generate_markdown(f"enhanced_compression_results_keep_bits_{self.keep_bits}_round_{self.round_off}.md")

if __name__ == "__main__":
    SIZE = 10000
    
    # Generate test data
    uniform_data = np.random.uniform(1e10, 1e12, SIZE)
    gaussian_data = np.random.normal(1e10, 1e12, SIZE)
    exponential_data = np.random.exponential(1e10, SIZE)

    all_data = {
        "uniform": uniform_data,
        "gaussian": gaussian_data,
        "exponential": exponential_data
    }
    
    pipeline3 = CompressionPipeline(size=SIZE, keep_bits=48, round_off=True)
    pipeline3.run_enhanced(all_data)
