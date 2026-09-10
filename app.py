import streamlit as st
import pandas as pd

from src.chat.assistant import process_ticket
from src.chat.chat_assistant import chat_with_assistant
from src.worknotes.worknotes import generate_work_notes
from src.recommender.recommender import generate_recommendation


st.set_page_config(
    page_title="OpsAssist AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 OpsAssist AI")
st.subheader("AI-Powered IT Operations & Service Desk Copilot")

st.sidebar.title("OpsAssist AI")

mode = st.sidebar.radio(
    "Select Mode",
    [
        "Ticket Analyzer",
        "IT Support Chat",
        "Model Benchmark"
    ]
)


# ============================================================
# TICKET ANALYZER
# ============================================================

if mode == "Ticket Analyzer":

    st.header("🎫 IT Ticket Analyzer")

    ticket_text = st.text_area(
        "Enter IT Support Ticket",
        placeholder="Example: I cannot access UKG on my new phone",
        height=150
    )

    if st.button("Analyze Ticket"):

        if ticket_text.strip():

            with st.spinner("Analyzing ticket..."):

                result = process_ticket(ticket_text)

                category = result["category"]
                ai_response = result["response"]

                work_notes = generate_work_notes(
                    ticket_text,
                    category,
                    ai_response
                )

                recommendation = generate_recommendation(
                    ticket_text,
                    category,
                    ai_response
                )

            st.subheader("🎯 Ticket Category")
            st.success(category)

            st.subheader("🤖 AI Support Response")
            st.write(ai_response)

            st.subheader("📝 ServiceNow Work Notes")
            st.code(
                work_notes,
                language="text"
            )

            st.subheader("💡 Resolution Recommendation")
            st.code(
                recommendation,
                language="text"
            )

        else:
            st.warning("Please enter a ticket.")


# ============================================================
# IT SUPPORT CHAT
# ============================================================

elif mode == "IT Support Chat":

    st.header("💬 IT Support Chat")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_message = st.chat_input(
        "Describe your IT issue..."
    )

    if user_message:

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_message
            }
        )

        with st.chat_message("user"):
            st.write(user_message)

        with st.chat_message("assistant"):

            with st.spinner("OpsAssist AI is thinking..."):

                conversation_history = [
                    {
                        "role": message["role"],
                        "content": message["content"]
                    }
                    for message in st.session_state.chat_history[:-1]
                ]

                response = chat_with_assistant(
                    user_message,
                    conversation_history
                )

            st.write(response)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": response
            }
        )


# ============================================================
# MODEL BENCHMARK
# ============================================================

else:

    st.header("📊 AI Model Benchmark")

    st.write(
        "Comparison of Llama 3.1, Llama 3.2 and Qwen 2.5 "
        "based on quality, speed, memory usage and overall score."
    )

    benchmark_file = "data/benchmark_results.csv"
    quality_file = "data/benchmark_quality_results.csv"
    scores_file = "data/model_scores.csv"

    try:

        benchmark_df = pd.read_csv(
            benchmark_file
        )

        quality_df = pd.read_csv(
            quality_file
        )

        scores_df = pd.read_csv(
            scores_file
        )

        winner = scores_df.iloc[0]

        st.subheader("🏆 Winning Model")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Best Model",
                winner["model"]
            )

        with col2:
            st.metric(
                "Overall Score",
                f"{winner['overall_score']}/100"
            )

        st.subheader("📋 Model Comparison")

        display_columns = [
            "model",
            "latency_seconds",
            "tokens_per_second",
            "peak_memory_mb",
            "quality_score",
            "speed_score",
            "memory_score",
            "overall_score",
            "rank"
        ]

        st.dataframe(
            scores_df[display_columns],
            use_container_width=True,
            hide_index=True
        )

        st.subheader("🎯 Quality Score")

        quality_chart = scores_df[
            ["model", "quality_score"]
        ].set_index("model")

        st.bar_chart(
            quality_chart
        )

        st.subheader("⚡ Speed")

        speed_chart = scores_df[
            ["model", "tokens_per_second"]
        ].set_index("model")

        st.bar_chart(
            speed_chart
        )

        st.subheader("💾 Memory Usage")

        memory_chart = scores_df[
            ["model", "peak_memory_mb"]
        ].set_index("model")

        st.bar_chart(
            memory_chart
        )

        st.subheader("🏅 Overall Model Score")

        overall_chart = scores_df[
            ["model", "overall_score"]
        ].set_index("model")

        st.bar_chart(
            overall_chart
        )

    except FileNotFoundError:

        st.error(
            "Benchmark results are not available. "
            "Please run the benchmark and scoring scripts first."
        )