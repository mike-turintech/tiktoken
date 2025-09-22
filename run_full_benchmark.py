#!/usr/bin/env python3
"""
Full benchmark script for tiktoken.
Downloads and uses a large text dataset for realistic benchmarking.
"""
import os
import sys
import time
import random
from pathlib import Path

# Set environment variable for thread count
if "RAYON_NUM_THREADS" not in os.environ:
    os.environ["RAYON_NUM_THREADS"] = "4"

# Add scripts to path to import benchmark function
sys.path.insert(0, str(Path(__file__).parent))
from scripts.benchmark import benchmark_batch

def download_sample_data():
    """Download or generate sample data for benchmarking with caching."""
    cache_dir = Path("benchmark_cache")
    cache_file = cache_dir / "benchmark_data.txt"

    # Check if cached data exists
    if cache_file.exists():
        print(f"Loading cached benchmark data from {cache_file}...")
        with open(cache_file, 'r', encoding='utf-8') as f:
            content = f.read()
        # Split back into documents (separated by special marker)
        documents = content.split('\n<<<DOCUMENT_SEPARATOR>>>\n')
        print(f"Loaded {len(documents)} cached documents ({len(content):,} bytes)")
        return documents

    try:
        import requests
        print("Downloading sample texts from Project Gutenberg...")

        # Extended list of public domain books for more comprehensive benchmarking
        urls = [
            ("https://www.gutenberg.org/files/1342/1342-0.txt", "Pride and Prejudice"),
            ("https://www.gutenberg.org/files/11/11-0.txt", "Alice in Wonderland"),
            ("https://www.gutenberg.org/files/84/84-0.txt", "Frankenstein"),
            ("https://www.gutenberg.org/files/1661/1661-0.txt", "Sherlock Holmes"),
            ("https://www.gutenberg.org/files/2701/2701-0.txt", "Moby Dick"),
            ("https://www.gutenberg.org/files/345/345-0.txt", "Dracula"),
            ("https://www.gutenberg.org/files/98/98-0.txt", "Tale of Two Cities"),
            ("https://www.gutenberg.org/files/1232/1232-0.txt", "The Prince"),
            ("https://www.gutenberg.org/files/2542/2542-0.txt", "Doll's House"),
            ("https://www.gutenberg.org/files/74/74-0.txt", "Tom Sawyer"),
            ("https://www.gutenberg.org/files/76/76-0.txt", "Huckleberry Finn"),
            ("https://www.gutenberg.org/files/1260/1260-0.txt", "Jane Eyre"),
            ("https://www.gutenberg.org/files/16/16-0.txt", "Peter Pan"),
            ("https://www.gutenberg.org/files/5200/5200-0.txt", "Metamorphosis"),
            ("https://www.gutenberg.org/files/219/219-0.txt", "Heart of Darkness"),
            # 15 additional books
            ("https://www.gutenberg.org/files/1952/1952-0.txt", "The Yellow Wallpaper"),
            ("https://www.gutenberg.org/files/844/844-0.txt", "The Importance of Being Earnest"),
            ("https://www.gutenberg.org/files/4300/4300-0.txt", "Ulysses"),
            ("https://www.gutenberg.org/files/2591/2591-0.txt", "Grimm's Fairy Tales"),
            ("https://www.gutenberg.org/files/1399/1399-0.txt", "Anna Karenina"),
            ("https://www.gutenberg.org/files/2600/2600-0.txt", "War and Peace"),
            ("https://www.gutenberg.org/files/1184/1184-0.txt", "The Count of Monte Cristo"),
            ("https://www.gutenberg.org/files/46/46-0.txt", "A Christmas Carol"),
            ("https://www.gutenberg.org/files/786/786-0.txt", "Twenty Thousand Leagues Under the Sea"),
            ("https://www.gutenberg.org/files/103/103-0.txt", "Around the World in Eighty Days"),
            ("https://www.gutenberg.org/files/1400/1400-0.txt", "Great Expectations"),
            ("https://www.gutenberg.org/files/730/730-0.txt", "Oliver Twist"),
            ("https://www.gutenberg.org/files/766/766-0.txt", "David Copperfield"),
            ("https://www.gutenberg.org/files/160/160-0.txt", "The Awakening"),
            ("https://www.gutenberg.org/files/174/174-0.txt", "The Picture of Dorian Gray"),
        ]

        texts = []
        total_size = 0
        for url, title in urls:
            try:
                print(f"  Downloading {title}...", end=" ")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                texts.append(response.text)
                size = len(response.text)
                total_size += size
                print(f"{size:,} bytes")
            except Exception as e:
                print(f"Failed: {e}")

        if texts:
            print(f"\nTotal downloaded: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")

            # Save to cache
            cache_dir.mkdir(exist_ok=True)
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write('\n<<<DOCUMENT_SEPARATOR>>>\n'.join(texts))
            print(f"Cached data saved to {cache_file}")

            return texts
    except ImportError:
        print("requests library not available, using generated data")

    # Fallback: generate synthetic data
    print("Generating synthetic benchmark data...")
    return generate_synthetic_data()

def generate_synthetic_data():
    """Generate synthetic text data for benchmarking."""
    samples = [
        "The quick brown fox jumps over the lazy dog. ",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ",
        "In the beginning was the Word, and the Word was with God. ",
        "To be or not to be, that is the question. ",
        "It was the best of times, it was the worst of times. ",
        "Call me Ishmael. Some years ago, never mind how long precisely. ",
        "It is a truth universally acknowledged that a single man must be in want of a wife. ",
        "All happy families are alike; each unhappy family is unhappy in its own way. ",
    ]

    # Generate ~10MB of text per document
    documents = []
    target_size = 10 * 1024 * 1024  # 10MB per document

    for i in range(10):  # Create 10 documents
        doc = []
        current_size = 0
        while current_size < target_size:
            text = random.choice(samples) * random.randint(10, 50)
            doc.append(text)
            current_size += len(text)
        documents.append(" ".join(doc))

    return documents

def run_benchmark_with_threads(documents, thread_counts=[2, 4, 8], iterations=10):
    """Run benchmark with different thread counts."""
    import tiktoken

    total_bytes = sum(len(doc.encode('utf-8')) for doc in documents)

    print("\n" + "=" * 70)
    print(f"TIKTOKEN PERFORMANCE BENCHMARK")
    print(f"Data size: {total_bytes:,} bytes ({total_bytes/1024/1024:.2f} MB)")
    print(f"Iterations: {iterations}")
    print("=" * 70)

    results = {}

    # Test tiktoken with different thread counts
    enc = tiktoken.get_encoding("o200k_base")  # GPT-5 encoding
    enc.encode("warmup")  # Warmup

    print("\nThread Count | Speed (bytes/s) | Speed (MB/s) | Time (ms)")
    print("-" * 60)

    for num_threads in thread_counts:
        os.environ["RAYON_NUM_THREADS"] = str(num_threads)

        # Run multiple iterations for better accuracy
        times = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            enc.encode_ordinary_batch(documents, num_threads=num_threads)
            end = time.perf_counter_ns()
            times.append(end - start)

        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        speed = total_bytes / avg_time * 1e9
        time_ms = avg_time / 1e6

        results[f"tiktoken_{num_threads}"] = {
            'speed_bytes': speed,
            'speed_mb': speed/1024/1024,
            'avg_time_ms': time_ms,
            'min_time_ms': min_time / 1e6,
            'max_time_ms': max_time / 1e6
        }

        print(f"{num_threads:^12} | {speed:>15,.0f} | {speed/1024/1024:>12.2f} | {time_ms:>9.2f}")

    # Show thread scaling analysis
    if len(thread_counts) > 1:
        print("\n" + "=" * 70)
        print("THREAD SCALING ANALYSIS")
        print("=" * 70)

        base_threads = thread_counts[0]
        base_speed = results[f"tiktoken_{base_threads}"]['speed_bytes']

        print(f"\nBaseline: {base_threads} thread(s) = {base_speed/1024/1024:.2f} MB/s")
        print("\nThread Count | Speedup | Efficiency | Time Reduction")
        print("-" * 60)

        for num_threads in thread_counts:
            speed = results[f"tiktoken_{num_threads}"]['speed_bytes']
            speedup = speed / base_speed
            efficiency = (speedup / (num_threads / base_threads)) * 100
            time_reduction = (1 - base_speed/speed) * 100

            print(f"{num_threads:^12} | {speedup:>7.2f}x | {efficiency:>10.1f}% | {time_reduction:>13.1f}%")

    # Show detailed timing statistics
    print("\n" + "=" * 70)
    print(f"DETAILED TIMING STATISTICS ({iterations} iterations)")
    print("=" * 70)

    print("\nThread Count | Avg (ms) | Min (ms) | Max (ms) | Std Dev")
    print("-" * 60)

    for num_threads in thread_counts:
        stats = results[f"tiktoken_{num_threads}"]
        variance = ((stats['max_time_ms'] - stats['min_time_ms']) / 2)  # Simple variance estimate

        print(f"{num_threads:^12} | {stats['avg_time_ms']:>8.2f} | {stats['min_time_ms']:>8.2f} | "
              f"{stats['max_time_ms']:>8.2f} | {variance:>8.2f}")

    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark tiktoken with different thread counts')
    parser.add_argument('--threads', nargs='+', type=int, default=[2, 4, 8],
                        help='Thread counts to test (default: 2 4 8)')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of iterations per thread count (default: 10)')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear cached benchmark data')
    parser.add_argument('--no-download', action='store_true',
                        help='Use synthetic data instead of downloading')
    args = parser.parse_args()

    print("=" * 70)
    print("TIKTOKEN MULTI-THREADED BENCHMARK")
    print("=" * 70)

    # Clear cache if requested
    if args.clear_cache:
        cache_file = Path("benchmark_cache/benchmark_data.txt")
        if cache_file.exists():
            cache_file.unlink()
            print("Cache cleared!")

    # Get data for benchmarking
    print("\nPreparing benchmark data...")
    if args.no_download:
        print("Using synthetic data (--no-download flag set)")
        documents = generate_synthetic_data()
    else:
        documents = download_sample_data()

    # Calculate total size
    total_bytes = sum(len(doc.encode('utf-8')) for doc in documents)
    print(f"\nTotal data size: {total_bytes:,} bytes ({total_bytes/1024/1024:.2f} MB)")
    print(f"Number of documents: {len(documents)}")

    # Run benchmarks with different thread counts
    thread_counts = args.threads
    print(f"Testing with thread counts: {thread_counts}")
    print(f"Iterations per thread count: {args.iterations}")

    try:
        results = run_benchmark_with_threads(documents, thread_counts, args.iterations)
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        print("\nMake sure you have tiktoken installed:")
        print("  pip install tiktoken")

        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()