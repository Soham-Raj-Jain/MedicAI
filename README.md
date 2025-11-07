# MedicAI 🏥

A powerful Retrieval-Augmented Generation (RAG) system designed specifically for medical document analysis. MedicAI combines semantic search with AI-powered generation to provide accurate, grounded answers from your medical literature.

## ✨ Key Features

- **Intelligent Document Indexing**: Automatically processes PDFs and CSVs, creating searchable embeddings using sentence-transformers
- **Semantic Search**: FAISS-powered vector similarity search for fast and accurate retrieval
- **Topic Classification**: Smart routing to relevant documents based on query context
- **Hybrid LLM Architecture**: 
  - Primary: GPT-4o-mini for high-quality responses
  - Fallback: Ollama for local inference
  - Extractive: Rule-based summarization when LLMs unavailable
- **Feedback-Driven Learning**: Continuous improvement through user feedback with automatic reranking
- **Citation Transparency**: Every answer includes source citations for medical traceability
- **Interactive UI**: Built with Gradio for easy deployment and sharing

## 🛠️ Technical Stack

- **Embeddings**: all-MiniLM-L6-v2 (384-dim)
- **Vector Database**: FAISS with cosine similarity
- **LLM Integration**: OpenAI API + Ollama support
- **Frontend**: Gradio
- **Document Processing**: PyPDF, Pandas
- **Python Libraries**: sentence-transformers, faiss-cpu, numpy, pandas

## 🚀 Getting Started

### Prerequisites

```bash
pip install streamlit plotly faiss-cpu sentence-transformers pypdf pandas numpy openai gradio
```

### Setup

1. **Configure API Keys** (optional):
```python
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
os.environ["OLLAMA_HOST"] = "http://localhost:11434"
os.environ["OLLAMA_MODEL"] = "phi3:mini"
```

2. **Organize Your Data**:
```
data/
├── Covid/
│   ├── document1.pdf
│   └── studies.csv
├── Diabetes/
│   ├── document2.pdf
│   └── studies.csv
└── Heart_Attack/
    ├── document3.pdf
    └── studies.csv
```

3. **Build Index**:
Run the indexing script to process your documents and create embeddings.

4. **Launch Application**:
```bash
python rag_gradio.py
```

## 📊 System Architecture

### Indexing Pipeline
1. Document extraction (PDF text, CSV schemas)
2. Text chunking (1000 chars with 120 char overlap)
3. Embedding generation (normalized 384-dim vectors)
4. FAISS index creation

### Retrieval Pipeline
1. Query embedding
2. Topic classification (semantic similarity)
3. Vector search with topic filtering
4. Feedback-based reranking
5. Cosine + feedback score fusion

### Generation Pipeline
1. Passage extraction from top-k results
2. LLM prompting with grounded context
3. Citation injection
4. Fallback handling (GPT → Ollama → Extractive)

## 🎯 Usage Example

```python
# Ask a question
query = "What are the prediction methods for heart disease using machine learning?"

# System automatically:
# 1. Classifies query → Heart_Attack topic
# 2. Retrieves relevant passages
# 3. Generates grounded answer with citations
# 4. Displays: Answer + Sources + Confidence scores
```

## 📈 Feedback System

MedicAI learns from user interactions:
- 👍 Thumbs up: Boosts document rankings (+15% weight)
- 👎 Thumbs down: Records poor results (-10% penalty)
- Persistent storage tracks chunk performance over time
- Reranking adapts based on historical feedback

## 🔒 Features & Highlights

- **Grounded Responses**: All answers derived from provided documents
- **Source Transparency**: Clear citations with page numbers
- **Scalable Design**: Handles large document collections efficiently
- **Cost-Efficient**: Automatic fallback prevents API failures
- **Privacy-Aware**: Can run fully local with Ollama
- **Domain-Adaptable**: Easy to retrain for different medical topics

## 📝 Configuration Options

- `SIM_THRESHOLD`: Minimum cosine similarity (default: 0.25)
- `MAX_PASSAGES`: Top passages for generation (default: 5)
- `EMB_NAME`: Sentence transformer model
- `EMB_DIM`: Embedding dimensions (384)


## 📄 License

This project is designed for educational and research purposes in medical document analysis.

## 🔍 How It Works

MedicAI operates in three stages:

1. **Preprocessing**: Documents are chunked and embedded into a semantic vector space
2. **Retrieval**: Queries are matched against document embeddings using cosine similarity
3. **Generation**: LLMs synthesize natural language answers from retrieved passages

The feedback loop continuously refines retrieval quality based on user satisfaction signals.

---

**Note**: This system requires pre-indexed medical documents to function. Ensure your data directory is properly structured before running the indexing pipeline.
