# Grounded Support Agent Demo

A standalone Streamlit demo for the Grounded Support Agent RAG pipeline. 
The pipeline uses a LangGraph router -> retriever -> generator -> critic flow with automatic escalation for out-of-scope or ungrounded queries.

## How to run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Export data artifacts from the source notebook using `export_artifacts.py`. Place them in the `data/` directory.
3. Set your Groq API key:
   ```bash
   export GROQ_API_KEY="your-api-key"
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Data Storage
The FAISS index, parquet chunks, and metadata are stored in the `data/` directory. Git LFS is configured for `.index` and `.parquet` files.

## Models
- **Live Demo**: Runs `llama-3.3-70b-versatile` on Groq for fast, accessible inference.
- **Evaluation Metrics**: The metrics displayed in the Eval tabs were measured on a local `Qwen2.5-7B-Instruct` (4-bit) model to match the original research environment.
