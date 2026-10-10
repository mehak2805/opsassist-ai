import streamlit as st
import pandas as pd
import plotly.express as px

from src.chat.assistant import process_ticket
from src.chat.chat_assistant import chat_with_assistant
from src.worknotes.worknotes import generate_work_notes
from src.recommender.recommender import generate_recommendation


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="OpsAssist AI",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# DARK UI
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background-color: #080b14;
        }

        [data-testid="stMetric"] {
            background-color: #111827;
            border: 1px solid #263653;
            padding: 18px;
            border-radius: 14px;
        }

        [data-testid="stMetricLabel"] {
            color: #9ca9c2;
        }

        [data-testid="stMetricValue"] {
            color: white;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.title("🤖 OpsAssist AI")
st.subheader("AI-Powered IT Operations & Service Desk Copilot")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🤖 OpsAssist AI")

mode = st.sidebar.radio(
    "Select Mode",
    [
        "Ticket Analyzer",
        "IT Support Chat",
        "Model Benchmark",
        "Analytics Dashboard"
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

    if st.button("Analyze Ticket", type="primary"):

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
            st.code(work_notes, language="text")

            st.subheader("💡 Resolution Recommendation")
            st.code(recommendation, language="text")

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

elif mode == "Model Benchmark":

    st.header("📊 AI Model Benchmark")

    st.write(
        "Comparison of Llama 3.1, Llama 3.2 and Qwen 2.5 "
        "based on quality, speed, memory usage and overall score."
    )

    scores_file = "data/model_scores.csv"

    try:

        scores_df = pd.read_csv(scores_file)

        winner = scores_df.iloc[0]

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "🏆 Best Model",
                winner["model"]
            )

        with col2:
            st.metric(
                "Overall Score",
                f"{winner['overall_score']:.2f}"
            )

        st.subheader("📋 Model Comparison")

        display_columns = [
            "model",
            "quality_score",
            "tokens_per_second",
            "peak_memory_mb",
            "overall_score",
            "rank"
        ]

        st.dataframe(
            scores_df[display_columns],
            use_container_width=True,
            hide_index=True
        )

        col1, col2 = st.columns(2)

        with col1:

            quality_fig = px.bar(
                scores_df,
                x="model",
                y="quality_score",
                title="Quality Score"
            )

            quality_fig.update_layout(
                template="plotly_dark"
            )

            st.plotly_chart(
                quality_fig,
                use_container_width=True
            )

        with col2:

            speed_fig = px.bar(
                scores_df,
                x="model",
                y="tokens_per_second",
                title="Speed — Tokens / Second"
            )

            speed_fig.update_layout(
                template="plotly_dark"
            )

            st.plotly_chart(
                speed_fig,
                use_container_width=True
            )

    except FileNotFoundError:

        st.error(
            "model_scores.csv was not found."
        )


# ============================================================
# ANALYTICS DASHBOARD
# ============================================================

else:

    st.title("📊 OpsAssist AI Analytics Dashboard")

    st.caption(
        "AI model performance and operational intelligence overview"
    )

    st.divider()

    try:

        scores_df = pd.read_csv(
            "data/model_scores.csv"
        )

        # ----------------------------------------------------
        # SORT MODELS
        # ----------------------------------------------------

        scores_df = scores_df.sort_values(
            "overall_score",
            ascending=False
        ).reset_index(drop=True)

        # ----------------------------------------------------
        # BASIC CALCULATIONS
        # ----------------------------------------------------

        total_models = len(scores_df)

        best_model = scores_df.iloc[0]["model"]

        best_score = float(
            scores_df.iloc[0]["overall_score"]
        )

        average_quality = float(
            scores_df["quality_score"].mean()
        )

        fastest_model = scores_df.loc[
            scores_df["tokens_per_second"].idxmax(),
            "model"
        ]

        lowest_memory_model = scores_df.loc[
            scores_df["peak_memory_mb"].idxmin(),
            "model"
        ]

        # ----------------------------------------------------
        # KPI CARDS
        # ----------------------------------------------------

        st.subheader("📈 Key Performance Indicators")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "AI Models Tested",
                total_models
            )

        with col2:
            st.metric(
                "🏆 Best Model",
                best_model
            )

        with col3:
            st.metric(
                "Overall Score",
                f"{best_score:.2f}"
            )

        with col4:
            st.metric(
                "Average Quality",
                f"{average_quality:.2f}"
            )

        with col5:
            st.metric(
                "⚡ Fastest Model",
                fastest_model
            )

        st.divider()

        # ----------------------------------------------------
        # MODEL PERFORMANCE SECTION
        # ----------------------------------------------------

        st.subheader("🏆 Model Performance")

        left, middle, right = st.columns([1.4, 1, 1])

        # ----------------------------------------------------
        # LEFT — RANKING
        # ----------------------------------------------------

        with left:

            st.markdown("### 🥇 AI Model Ranking")

            ranking_df = scores_df[
                [
                    "model",
                    "quality_score",
                    "overall_score",
                    "rank"
                ]
            ].copy()

            ranking_df["rank"] = ranking_df["rank"].astype(int)

            st.dataframe(
                ranking_df,
                use_container_width=True,
                hide_index=True
            )

        # ----------------------------------------------------
        # MIDDLE — QUALITY
        # ----------------------------------------------------

        with middle:

            st.markdown("### 🎯 Quality")

            quality_fig = px.bar(
                scores_df,
                x="model",
                y="quality_score",
                text="quality_score"
            )

            quality_fig.update_traces(
                texttemplate="%{text:.1f}",
                textposition="outside"
            )

            quality_fig.update_layout(
                template="plotly_dark",
                height=350,
                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10
                ),
                xaxis_title="",
                yaxis_title=""
            )

            st.plotly_chart(
                quality_fig,
                use_container_width=True
            )

        # ----------------------------------------------------
        # RIGHT — SPEED
        # ----------------------------------------------------

        with right:

            st.markdown("### ⚡ Speed")

            speed_fig = px.bar(
                scores_df,
                x="model",
                y="tokens_per_second",
                text="tokens_per_second"
            )

            speed_fig.update_traces(
                texttemplate="%{text:.1f}",
                textposition="outside"
            )

            speed_fig.update_layout(
                template="plotly_dark",
                height=350,
                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10
                ),
                xaxis_title="",
                yaxis_title=""
            )

            st.plotly_chart(
                speed_fig,
                use_container_width=True
            )

        # ----------------------------------------------------
        # SECOND ROW
        # ----------------------------------------------------

        st.subheader("📊 Performance Analysis")

        col1, col2 = st.columns(2)

        # ----------------------------------------------------
        # MEMORY
        # ----------------------------------------------------

        with col1:

            st.markdown("### 💾 Memory Usage")

            memory_fig = px.bar(
                scores_df,
                x="model",
                y="peak_memory_mb",
                text="peak_memory_mb"
            )

            memory_fig.update_traces(
                texttemplate="%{text:.1f} MB",
                textposition="outside"
            )

            memory_fig.update_layout(
                template="plotly_dark",
                height=400,
                xaxis_title="AI Model",
                yaxis_title="Memory (MB)"
            )

            st.plotly_chart(
                memory_fig,
                use_container_width=True
            )

        # ----------------------------------------------------
        # OVERALL SCORE
        # ----------------------------------------------------

        with col2:

            st.markdown("### 🏅 Overall Model Score")

            overall_fig = px.bar(
                scores_df,
                x="model",
                y="overall_score",
                text="overall_score"
            )

            overall_fig.update_traces(
                texttemplate="%{text:.2f}",
                textposition="outside"
            )

            overall_fig.update_layout(
                template="plotly_dark",
                height=400,
                xaxis_title="AI Model",
                yaxis_title="Overall Score"
            )

            st.plotly_chart(
                overall_fig,
                use_container_width=True
            )

        # ----------------------------------------------------
        # AI INSIGHTS
        # ----------------------------------------------------

        st.subheader("🤖 AI Performance Insights")

        insight1, insight2, insight3 = st.columns(3)

        with insight1:

            st.info(
                f"🏆 **Best Overall**\n\n"
                f"{best_model} achieved the highest "
                f"overall score of **{best_score:.2f}**."
            )

        with insight2:

            fastest_speed = scores_df[
                "tokens_per_second"
            ].max()

            st.success(
                f"⚡ **Fastest Model**\n\n"
                f"{fastest_model} generated "
                f"**{fastest_speed:.2f} tokens/sec**."
            )

        with insight3:

            lowest_memory = scores_df[
                "peak_memory_mb"
            ].min()

            st.warning(
                f"💾 **Most Memory Efficient**\n\n"
                f"{lowest_memory_model} used "
                f"**{lowest_memory:.2f} MB** peak memory."
            )

        # ----------------------------------------------------
        # FINAL COMPARISON TABLE
        # ----------------------------------------------------

        st.subheader("📋 Complete Model Comparison")

        st.dataframe(
            scores_df[
                [
                    "model",
                    "quality_score",
                    "tokens_per_second",
                    "peak_memory_mb",
                    "overall_score",
                    "rank"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    except FileNotFoundError:

        st.error(
            "❌ data/model_scores.csv was not found. "
            "Please run your benchmark and scoring scripts first."
        )

    except Exception as e:

        st.error(
            f"❌ Dashboard error: {e}"
        )