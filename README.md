🤖 OpsAssist AI
AI-Powered IT Operations & Service Desk Copilot
OpsAssist AI is an AI-powered IT Operations and Service Desk Copilot designed to assist IT support teams with ticket classification, knowledge retrieval, troubleshooting guidance, ServiceNow work-note generation, resolution recommendations, conversational support, and local LLM benchmarking.
The project combines Machine Learning, NLP, Retrieval-Augmented Generation (RAG), and locally hosted Large Language Models to build an end-to-end IT support assistant.
🚀 Features
🎫 IT Ticket Classification

Automatically classifies IT support tickets into relevant categories using:
TF-IDF Vectorization
Logistic Regression
Scikit-learn

Supported ticket categories include:
Login Issue
Password Reset
Account Locked
Application Access
Hardware Issue
Authentication Issue
Email Issue
VPN Issue
Software Request
Software Issue
Access Request
Portal Issue
📚 RAG-Based Knowledge Retrieval
OpsAssist retrieves relevant information from an internal IT Knowledge Base before generating an AI response.

Current Knowledge Base examples:
UKG Application Access
Password Reset
The retrieval system uses:
TF-IDF
Cosine Similarity
Python
Markdown-based Knowledge Documents
This helps keep AI responses grounded in available IT support information.

🤖 AI Support Response
OpsAssist uses locally hosted LLMs through Ollama to generate concise IT support guidance.
The response provides:
Issue description
Troubleshooting steps
Recommended action
Escalation decision
The AI is instructed not to invent unsupported procedures, URLs, access codes, policies, or technical details.

📝 ServiceNow Work Notes Generator
OpsAssist automatically converts ticket information and resolution details into professional ServiceNow-style work notes.
Generated sections:
Reason for call
Resolution
Action items
Details
The system is designed to avoid presenting recommended actions as completed actions.
💡 Resolution Recommendation
The system analyzes the ticket and AI support response to provide:
Recommended Resolution
Next Action
Escalation
Escalation Reason
This helps IT support agents determine the appropriate next step.

💬 IT Support Chat
OpsAssist includes a conversational IT support assistant with:
Multi-turn conversations
Conversation history
Knowledge Base retrieval
Context-aware responses
IT troubleshooting guidance

📊 Local LLM Benchmarking
OpsAssist benchmarks three locally hosted models:
Llama 3.1
Llama 3.2
Qwen 2.5
The benchmark measures:
Response latency
Tokens per second
Memory usage
Response quality
Overall model score
🏆 Model Benchmark Results
The current local benchmark produced the following results:
Model	Quality Score	Overall Score	Rank
Qwen 2.5	50.83	58.23	1
Llama 3.2	28.33	54.21	2
Llama 3.1	41.67	45.00	3
🥇 Current Winning Model
Qwen 2.5 — Overall Score: 58.23/100
These results are specific to the local hardware, models, prompts, and evaluation dataset used in this project and should not be interpreted as universal model rankings.
🧠 System Architecture
User
↓
Streamlit Application
↓
Ticket Analyzer / IT Support Chat / Model Benchmark
↓
ML Classifier + RAG Pipeline + Ollama LLM
↓
AI Support Response
↓
ServiceNow Work Notes + Resolution Recommendation
🛠️ Technology Stack
Programming
Python
SQL
C++
Machine Learning and NLP
Scikit-learn
TF-IDF
Logistic Regression
Cosine Similarity
NLP
Generative AI
Ollama
Llama 3.1
Llama 3.2
Qwen 2.5
Retrieval-Augmented Generation
Data and Application
Pandas
NumPy
Streamlit
Development Tools
VS Code
Git
GitHub
Ollama

📂 Project Structure
opsassist_ai/
├── data/
│ ├── raw/
│ ├── processed/
│ ├── knowledge_base/
│ │ ├── loader.py
│ │ ├── password_reset.md
│ │ └── ukg_access.md
│ │
│ ├── tickets.csv
│ │
│ └── benchmark/
│ ├── evaluation_cases.csv
│ └── benchmark_quality_results.csv
│
├── src/
│ ├── classifier/
│ │ └── classifier.py
│ │
│ ├── chat/
│ │ ├── assistant.py
│ │ └── chat_assistant.py
│ │
│ ├── rag/
│ │ ├── retriever.py
│ │ └── rag_pipeline.py
│ │
│ ├── recommender/
│ │ └── recommender.py
│ │
│ ├── worknotes/
│ │ └── worknotes.py
│ │
│ └── benchmark/
│ ├── benchmark.py
│ ├── evaluator.py
│ └── scoring.py
│
├── tests/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

⚙️ Installation
1. Clone the repository
Clone this repository from GitHub and open the project folder in VS Code.
2. Create a virtual environment
python3 -m venv .venv
3. Activate the virtual environment
For macOS and Linux:
source .venv/bin/activate
For Windows:
.venv\Scripts\activate
4. Install Python dependencies
pip install -r requirements.txt
5. Install Ollama Models
Make sure Ollama is installed and running.
Pull the required models:
ollama pull llama3.1
ollama pull llama3.2
ollama pull qwen2.5

▶️ Run the Application
From the project root directory:
python -m streamlit run app.py
The application provides three modes.
Ticket Analyzer
Analyze an IT support ticket and generate:
Ticket category
AI support response
ServiceNow work notes
Resolution recommendation
IT Support Chat
Interact with OpsAssist through a multi-turn IT support conversation.

Model Benchmark
View model performance and comparison results.

📊 Run Model Benchmark
Run the performance benchmark:
python src/benchmark/benchmark.py
This generates:
data/benchmark_results.csv
Run the quality evaluation:
python src/benchmark/evaluator.py
This generates:
data/benchmark_quality_results.csv
Calculate the final model scores:
python src/benchmark/scoring.py
This generates:
data/model_scores.csv
🔬 Evaluation Methodology
OpsAssist evaluates local LLMs using multiple performance metrics.
Response Quality
Measures how many expected resolution points are covered by the model response.
Speed
Speed is measured using the number of generated tokens divided by response latency.
Memory
Memory usage is measured during the model benchmark.
Overall Score
The current scoring formula is:
Quality Score × 50%
Speed Score × 30%
Memory Efficiency Score × 20%
The model with the highest overall score is selected as the benchmark winner.
🔐 Design Principles
OpsAssist AI follows these principles:
Prefer grounded responses over hallucinated information.
Use relevant Knowledge Base information.
Avoid inventing IT procedures.
Avoid unsupported URLs, access codes, and policies.
Clearly identify when escalation may be required.
Keep ServiceNow work notes concise and professional.
Support local LLM execution through Ollama.
🎯 Use Case
OpsAssist AI is designed for IT Service Desk and IT Operations environments where support teams frequently handle repetitive issues such as:
Password problems
Application access
Authentication issues
VPN issues
Email problems
Hardware issues
Portal access
Software requests
The system can help reduce repetitive manual work and provide more consistent support guidance.


🔮 Future Improvements
Possible future enhancements include:
Larger real-world ticket datasets
Semantic embeddings
Vector database integration
Advanced RAG pipelines
Improved response evaluation
ServiceNow API integration
Automated ticket routing
Role-based access
Production monitoring
Fine-tuned IT support models


👩‍💻 Author
Mankirat Kaur
MCA — Artificial Intelligence and Machine Learning
Associate – IT Operations
Skills
Python
SQL
Machine Learning
NLP
Generative AI
RAG
Streamlit
Data Analysis
IT Operations



⚠️ Disclaimer
This project is an educational and portfolio implementation.
The benchmark results and generated IT recommendations are based on local models, synthetic evaluation data, and the configured Knowledge Base.
The generated recommendations should be validated against official organizational procedures before being used in a production IT environment.