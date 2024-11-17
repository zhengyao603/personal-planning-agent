import whisper
import gradio as gr

from component.agent import SchedulePlanningAgent

def agent_chat(user_input, chat_history):
    transcription_model = whisper.load_model("base")
    transcription_result = transcription_model.transcribe(user_input)
    print(transcription_result["text"])

    # TODO: maybe manage chat history with some external storage

    planning_agent = SchedulePlanningAgent()
    planning_agent_response =planning_agent.full_chain.invoke({"input": transcription_result["text"]})["output"]
    chat_history.append((transcription_result["text"], planning_agent_response))
    return chat_history

def main() -> None:

    with gr.Blocks() as app:
        gr.Markdown("## Personal Planning Agent😊")
        chat_box = gr.Chatbot(label="chatbot", show_label=False)
        user_input = gr.Audio(sources=["microphone"], type="filepath")
        send_button = gr.Button("Send")

        send_button.click(
            fn=agent_chat,
            inputs=[user_input, chat_box],
            outputs=[chat_box],
        )
    app.launch()

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    main()