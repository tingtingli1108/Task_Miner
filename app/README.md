# TaskMiner

**Automatically extract actionable tasks from email using AI**

TaskMiner uses a RAG (Retrieval-Augmented Generation) pipeline to intelligently identify and extract tasks from email files, helping you stay organized and never miss important action items.

## ✨ Features

- **Multi-Provider AI Support**: Works with OpenAI (e.g. GPT-4, o3-mini) and Anthropic (Claude) models
- **Smart Task Detection**: Identifies actionable items requiring your attention
- **Context-Aware Extraction**: Uses email history for better task understanding  
- **Web Dashboard**: User-friendly interface for reviewing and managing extracted tasks
- **Batch Processing**: Upload and process multiple emails at once

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- AWS account (DynamoDB, S3)
- API keys for OpenAI and/or Anthropic

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Capstone_TaskMiner.git
   cd Capstone_TaskMiner
   ```

2. **Install dependencies**
   ```bash
   cd app
   pip install poetry
   poetry install
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and AWS configuration
   ```

4. **Run the application**
   ```bash
   streamlit run Home.py
   ```

## 📋 Usage

1. **Upload Email Files**: Upload `.eml` files through the web interface
2. **AI Processing**: TaskMiner automatically detects and extracts tasks
3. **Review Results**: Approve, edit, or decline extracted tasks
4. **Stay Organized**: Track your tasks and action items in one place

## 🧪 Evaluation Mode

For testing and evaluation without database modifications:

```python
from core.RAG_database import rag_pipeline

# Run in evaluation mode (no database changes)
result = rag_pipeline(
    user_id="test@example.com",
    user_name="Test User", 
    email_id="test_email_123",
    model_id="openai:gpt-4.1",
    detection_prompt_variant="production_baseline",
    extraction_prompt_variant="production_baseline",
    evaluation_mode=True  # Skip database modifications
)

# Access comprehensive results
print(f"Task detected: {result['pipeline_outputs']['pred_has_task']}")
print(f"Tasks extracted: {len(result['pipeline_outputs']['extracted_tasks'])}")
print(f"Retrieved context IDs: {result['pipeline_outputs']['pred_retrieved_ids']}")
```

**Benefits of evaluation mode:**
- 🚫 **No database pollution** - Perfect for testing different models/prompts
- ⚡ **Faster execution** - Skips time-consuming database writes
- 🔬 **Clean metrics** - Get accurate performance data
- 🔄 **Repeatable runs** - Test the same emails multiple times

## 🔧 Configuration

Set these environment variables in your `.env` file:

```bash
# LLM API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# AWS Configuration  
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
S3_BUCKET_NAME=your_bucket_name

# Vector Store
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=your_index_name
```

## 🧪 Testing

Run the test suite to verify everything works:

```bash
cd app
pytest test/ -v
```

Test specific models:
```bash
pytest test/full_pipeline_test.py::test_rag_pipeline_with_model -v
```

## 🏗️ Architecture

- **Frontend**: Streamlit web application
- **Backend**: Python with LangChain for LLM integration
- **Storage**: AWS DynamoDB for tasks/emails, S3 for file storage
- **Vector Store**: Pinecone for semantic search and context retrieval
- **AI Models**: OpenAI GPT models and Anthropic Claude models

## 📖 Documentation

- **Detailed Setup**: See `app/PROJECT_PLAN.md` for comprehensive documentation
- **Prompt Management**: See `app/prompts/README.md` for prompt customization
- **API Reference**: Core modules documented in `app/core/`


## 💰 Pricing

- **OpenAI(ChatGPT):** https://platform.openai.com/docs/pricing
- **Anthropic(Claude):** https://docs.anthropic.com/en/docs/about-claude/pricing


## 👥 Team

This project was created by the UC Berkeley MIDS (Master of Information and Data Science) Capstone Team:

- **Tingting Li**
- **Sebastian Rosales**
- **Safi Aharoni**
- **Peng Zhao**
- **David Dorfman**

Thank you to our professors, classmates, and SMEs for the invaluable feedback throughout this project.