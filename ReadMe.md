# RAG Application from Scratch

A modular Retrieval-Augmented Generation (RAG) pipeline built from scratch, wrapped in a simple Gradio app to demonstrate each phase:

* **Ingesting Phase**: Chunk and embed source data, store in a vector database.
* **Retrieval Phase**: Embed user queries, perform similarity search on stored embeddings.
* **LLM Inference**: Generate responses using an LLM, combining context with the user prompt.
* **App**: Expose the pipeline via a Gradio interface.

---

## Architecture

The following diagram illustrates the flow of data through the RAG pipeline:

**Flowchart**: Place your pipeline diagram at the project root as `flowchart.png`, then embed it inline:

```markdown
![RAG Pipeline Flowchart](flowchart.png)
```

---

## Project Structure

```
RAG_From_Scratch/
├── .gitignore         # ignore env, IDE configs, logs, build artifacts, raw data
├── Requirements.txt   # pinned Python dependencies
├── notebooks/
│   └── Rag_test.ipynb # exploratory notebook for pipeline tests
├── rag_app/
│   └── App.py         # Gradio application tying all modules together
└── src/
    ├── Ingesting_phase.py   # chunking, embedding, and storage logic
    ├── Retrival_phase.py    # query embedding and vector DB search
    └── LLM_Inference.py     # LLM prompt construction and completion logic
```

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your_username/RAG_From_Scratch.git
   cd RAG_From_Scratch
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv env
   source env/bin/activate    # macOS/Linux
   env\Scripts\activate     # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r Requirements.txt
   ```

---

## Usage

1. **Prepare your data**: Place or point your source documents in the appropriate folder or update `Ingesting_phase.py` to load from your data source.

2. **Run ingestion** (optional manual test):

   ```bash
   python src/Ingesting_phase.py
   ```

3. **Test retrieval & inference** via notebook:

   ```bash
   jupyter notebook notebooks/Rag_test.ipynb
   ```

4. **Launch the Gradio app**:

   ```bash
   python rag_app/App.py
   ```

5. **Interact in your browser**: Open the local URL printed in your console to enter queries and see generated responses.

---

## Module Details

* **Ingesting\_phase.py**:

  * Splits raw documents into chunks.
  * Generates embeddings using your chosen model (CoHere, OpenAI, etc.).
  * Stores embeddings in a vector database (e.g., Chroma, FAISS).

* **Retrival\_phase.py**:

  * Converts user input into embeddings.
  * Performs similarity search against stored embeddings.
  * Returns top-k relevant chunks.

* **LLM\_Inference.py**:

  * Constructs a prompt combining user query and retrieved context.
  * Calls an LLM (GPT-4, Claude, etc.) to generate the final answer.

* **App.py**:

  * Integrates all phases into a Gradio interface.
  * Handles user input, displays results, and allows simple configuration.

---

## Contributing

Feel free to open issues or submit pull requests! For major changes, please open an issue first to discuss what you’d like to change.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
