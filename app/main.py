import os
import nltk  # for data preprocessing
import requests
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv

# Import your RAG package components
from rag_builder.Ingesting_phase import DocumentLoader
from rag_builder.Retrival_phase import dv, reset_database
from rag_builder.LLM_Inference import get_response

# Download necessary NLTK data
nltk.download("punkt")

# Load environment variables (SECRET_API_KEY should be set there)
# By default, load_dotenv() loads a .env file from the current working directory
load_dotenv()

# Gradio application logic
def run_app(file_obj, url_input, user_query):
    # Clear out any previous state
    reset_database()

    # Ingest document or URL
    if url_input:
        html = requests.get(url_input).text
        temp_path = Path("./temp_url.html")
        temp_path.write_text(html, encoding="utf-8")
        loader = DocumentLoader(str(temp_path))
        orig_chunks, proc_chunks = loader.load_html()
        dv.original_docs.extend(orig_chunks)
        dv.add_documents(proc_chunks)
        temp_path.unlink()
    elif file_obj:
        ext = Path(file_obj.name).suffix.lower().lstrip('.')
        loader = DocumentLoader(file_obj.name)
        if ext == 'pdf':
            orig_chunks, proc_chunks = loader.load_pdf()
        elif ext == 'txt':
            orig_chunks, proc_chunks = loader.load_text()
        else:
            return "Unsupported file type.\nPlease upload PDF or TXT.", ""
        dv.original_docs.extend(orig_chunks)
        dv.add_documents(proc_chunks)
    else:
        return "Please upload a file or enter a URL.", ""

    # Base model output (no context)
    base_output = get_response(user_query, "")

            # RAG-enhanced output: gather best matches as context
    matches = dv.find_best_matches(user_query)
    flat_context = []
    for m in matches:
        if isinstance(m, list):
            flat_context.extend(m)
        else:
            flat_context.append(m)
    context = "".join(flat_context)
    rag_output = get_response(user_query, context)

    return base_output, rag_output

# Build and launch Gradio UI
def main():
    with gr.Blocks() as demo:
        gr.Markdown("## RAG vs. Base Model Comparison")
        with gr.Row():
            file_input = gr.File(label="Upload PDF/TXT", file_types=[".pdf", ".txt"])
            url_input = gr.Textbox(label="Or enter HTML URL", placeholder="https://...")
        query_input = gr.Textbox(label="Ask a question:")
        run_btn = gr.Button("Run")
        out_base = gr.Textbox(label="Base Model Output", lines=5)
        out_rag = gr.Textbox(label="RAG-Enhanced Output", lines=5)

        run_btn.click(fn=run_app,
                      inputs=[file_input, url_input, query_input],
                      outputs=[out_base, out_rag])

    demo.launch(share= True)

if __name__ == "__main__":
    main()
    