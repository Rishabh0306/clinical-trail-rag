import gradio as gr
from clinical_trail_rag import ClinicalTrialRAG

ct_rag = ClinicalTrialRAG()
ct_rag.initialize_rag_pipeline()


def generate_response(message, history):
    return str(ct_rag.generate_response(message))


io = gr.ChatInterface(generate_response,
                      chatbot=gr.Chatbot(height=200),
                      textbox=gr.Textbox(placeholder="Ask a question related to clinical trails"),
                      title="Clinical Trail RAG",
                      examples=["Which company has conducted trail for BIBF 1120?"],
                      undo_btn=None,
                      retry_btn=None,

                      )

if __name__ == '__main__':
    io.launch(server_name="0.0.0.0", server_port=9090)
