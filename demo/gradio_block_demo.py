import gradio as gr
import whisper
import pyttsx3
import tempfile

try:
    engine = pyttsx3.init()
except Exception as e:
    print(f"Failed to initialize TTS engine: {e}")
    engine = None

def text_to_speech(text):
    # 将文本转换为语音
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_file_name = temp_file.name
    temp_file.close()

    engine.save_to_file(text, temp_file_name)
    engine.runAndWait()

    return temp_file_name


def chatbot(user_input, chat_history):
    print(user_input)
    transcription_model = whisper.load_model("base")
    transcription_result = transcription_model.transcribe(user_input)
    print(transcription_result["text"])

    # 忽略输入，始终返回 "hello world"
    response_text = "hello world"
    chat_history.append((transcription_result["text"], response_text))
    response_audio = text_to_speech(response_text)
    print(response_audio)

    return chat_history, response_audio


# 创建 Gradio 接口
with gr.Blocks() as demo:
    gr.Markdown("## Personal Planning Agent😊")
    chat_box = gr.Chatbot(label="chatbot", show_label=False)

    audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Input Audio")
    audio_output = gr.Audio(label="Response Audio")

    submit_button = gr.Button("Submit")
    submit_button.click(
        fn=chatbot,
        inputs=[audio_input, chat_box],
        outputs=[chat_box, audio_output]
    )

# 启动接口
demo.launch()