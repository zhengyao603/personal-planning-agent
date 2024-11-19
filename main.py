import whisper
import pyttsx3
import tempfile
import gradio as gr

from component.agent import PlanningAgent

def audio_to_text(audio_input):
    transcription_model = whisper.load_model("base")
    transcription_result = transcription_model.transcribe(audio_input)
    return transcription_result["text"]

def agent_response(text_input):
    planning_agent = PlanningAgent()
    planning_agent_response = planning_agent.full_chain.invoke({"input": text_input})["output"]
    return planning_agent_response

def text_to_audio(text_input):
    engine = pyttsx3.init()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_file_name = temp_file.name
    temp_file.close()

    engine.save_to_file(text_input, temp_file_name)
    engine.runAndWait()

    return temp_file_name

def agent_chat(user_input, chat_history):
    transcription_result = audio_to_text(user_input)
    text_response = agent_response(transcription_result)
    audio_response = text_to_audio(text_response)

    chat_history.append((transcription_result, text_response))
    return chat_history, audio_response

def main():
    with gr.Blocks() as app:
        gr.Markdown("## Personal Planning Agent😊")
        chat_box = gr.Chatbot(label="ChatBot", show_label=False)
        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Audio Input")
        audio_output = gr.Audio(label="Audio Output")

        send_button = gr.Button("Send")

        send_button.click(
            fn=agent_chat,
            inputs=[audio_input, chat_box],
            outputs=[chat_box, audio_output],
        )
    app.launch()

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    main()