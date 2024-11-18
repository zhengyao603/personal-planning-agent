import whisper
import gradio as gr

def agent_chat(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result["text"]

def main():
    audio_input = gr.Audio(
        sources=["microphone"],
        type="filepath",
    )
    app = gr.Interface(
        fn=agent_chat,
        inputs=[audio_input],
        outputs=["text"],
    )
    app.launch(share=True)

if __name__ == "__main__":
    main()