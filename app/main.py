##importing relevant libraries and modules 
import os
import nltk  
import requests
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv

# Importing my personal rag packages and modules
from rag_builder.Ingesting_phase import DocumentLoader
from rag_builder.Retrival_phase import dv, reset_database
from rag_builder.LLM_Inference import get_response


nltk.download("punkt")


#this is to load the env 
load_dotenv()

# buidling the gradio logic
def run_app(file_obj, url_input, user_query):
    # Clearing out any previous input
    reset_database()

    # handling the ingestion
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

    # Base model output to handle cases with no context
    base_output = get_response(user_query, "")

            ##gathering the best matches as context
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

# buidling the gradio interface
def main():
    with gr.Blocks() as demo:
        gr.Markdown("## RAG vs. Base Model Comparison: Kindly Provide A Document or A Link And Ask Questions")
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
