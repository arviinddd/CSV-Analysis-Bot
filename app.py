import gradio as gr
from utils.process_files import ProcessFiles
from utils.chatbot import CSVAnalysisBot

bot = CSVAnalysisBot()

def process_and_initialize(file):
    """Process the CSV file and prepare the system for queries."""
    processor = ProcessFiles(file.name)
    processor.process_and_store()
    return f"File {file.name} processed successfully."

def query_bot(message, file_name):
    """Handle user queries by automatically determining query type."""
    return bot.respond(message, file_name)

with gr.Blocks() as demo:
    with gr.TabItem("CSV Analysis Bot"):
        file_input = gr.File(label="Upload CSV File")
        chat_input = gr.Textbox(label="Ask a Question")
        response_output = gr.Textbox(label="Response")

        file_input.upload(process_and_initialize, inputs=[file_input], outputs=[response_output])

        chat_input.submit(query_bot, inputs=[chat_input, file_input], outputs=[response_output])

if __name__ == "__main__":
    demo.launch()