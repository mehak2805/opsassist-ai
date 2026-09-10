import time
import pandas as pd
import ollama
import psutil


# Models to benchmark
MODELS = [
    "llama3.1",
    "llama3.2",
    "qwen2.5"
]


# Same prompt for every model
PROMPT = """
You are an IT Service Desk Copilot.

A user has reported the following IT issue:

I cannot access UKG on my new phone.

Provide a concise and professional IT support response.

Include:
1. Likely issue
2. Troubleshooting steps
3. Recommended action
4. Escalation if required

Use only information relevant to the issue.
Do not invent URLs, access codes, policies, or technical details.
"""


def get_memory_mb():
    """
    Get current Python process memory usage in MB.
    """
    process = psutil.Process()

    memory_bytes = process.memory_info().rss

    memory_mb = memory_bytes / (1024 * 1024)

    return memory_mb


def benchmark_model(model):

    print(f"\nBenchmarking {model}...")

    # Measure memory before request
    memory_before = get_memory_mb()

    # Start timer
    start_time = time.perf_counter()

    # Send request to Ollama
    response = ollama.generate(
        model=model,
        prompt=PROMPT
    )

    # End timer
    end_time = time.perf_counter()

    # Measure memory after request
    memory_after = get_memory_mb()

    # Calculate latency
    latency = end_time - start_time

    # Get generated token count
    tokens_generated = response.get(
        "eval_count",
        0
    )

    # Calculate tokens per second
    if latency > 0:
        tokens_per_second = (
            tokens_generated / latency
        )
    else:
        tokens_per_second = 0

    # Calculate memory usage
    peak_memory = max(
        memory_before,
        memory_after
    )

    print(
        f"{model}: "
        f"{latency:.3f}s | "
        f"{tokens_per_second:.2f} tok/s | "
        f"{peak_memory:.2f} MB"
    )

    return {
        "model": model,
        "latency_seconds": round(
            latency,
            3
        ),
        "tokens_generated": tokens_generated,
        "tokens_per_second": round(
            tokens_per_second,
            2
        ),
        "peak_memory_mb": round(
            peak_memory,
            2
        )
    }


def run_benchmark():

    results = []

    for model in MODELS:

        try:

            result = benchmark_model(
                model
            )

            results.append(result)

        except Exception as e:

            print(
                f"Error benchmarking "
                f"{model}: {e}"
            )


    if not results:

        print(
            "\nNo benchmark results "
            "were generated."
        )

        return


    # Convert results to DataFrame
    df = pd.DataFrame(results)


    # Display benchmark table
    print("\n" + "=" * 70)

    print(
        "OPSASSIST AI MODEL BENCHMARK"
    )

    print("=" * 70)


    print(
        df[
            [
                "model",
                "latency_seconds",
                "tokens_generated",
                "tokens_per_second",
                "peak_memory_mb"
            ]
        ].to_string(index=False)
    )


    # Save benchmark results
    df.to_csv(
        "data/benchmark_results.csv",
        index=False
    )


    print(
        "\nResults saved to:"
        " data/benchmark_results.csv"
    )


if __name__ == "__main__":

    run_benchmark()