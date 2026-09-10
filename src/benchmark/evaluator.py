import pandas as pd
import re
import ollama


MODELS = [
    "llama3.1",
    "llama3.2",
    "qwen2.5"
]


def normalize_text(text):
    return re.sub(r"[^a-z0-9\s]", " ", text.lower())


def calculate_quality_score(response, expected_points):

    response_text = normalize_text(response)

    points = [
        point.strip()
        for point in expected_points.split(";")
        if point.strip()
    ]

    matched_points = 0

    for point in points:

        keywords = normalize_text(point).split()

        if all(keyword in response_text for keyword in keywords):
            matched_points += 1

    if not points:
        return 0.0

    score = (matched_points / len(points)) * 100

    return round(score, 2)


def generate_model_response(model, ticket):

    prompt = f"""
You are an IT Service Desk Copilot.

Provide a concise and professional troubleshooting response
for the following IT ticket:

{ticket}

Include:
- Likely issue
- Troubleshooting steps
- Recommended action
- Escalation if required

Do not invent technical procedures.
"""

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


def evaluate_models():

    df = pd.read_csv(
        "data/benchmark/evaluation_cases.csv"
    )

    results = []

    for model in MODELS:

        print(f"\nEvaluating {model}...")

        total_score = 0

        for index, row in df.iterrows():

            response = generate_model_response(
                model,
                row["ticket"]
            )

            score = calculate_quality_score(
                response,
                row["expected_points"]
            )

            total_score += score

            print(
                f"Case {index + 1}: "
                f"{score}%"
            )

        average_score = total_score / len(df)

        results.append(
            {
                "model": model,
                "quality_score": round(
                    average_score,
                    2
                )
            }
        )

        print(
            f"{model} Average Quality: "
            f"{average_score:.2f}%"
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "data/benchmark_quality_results.csv",
        index=False
    )

    print("\n" + "=" * 50)
    print("MODEL QUALITY BENCHMARK")
    print("=" * 50)

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        "\nSaved to:"
        " data/benchmark_quality_results.csv"
    )


if __name__ == "__main__":

    evaluate_models()