import gradio as gr

def chat_with_ai(user_input, chat_history):
    ai_response = f"AI: {user_input[::-1]}"  # 示例：简单地反转用户输入作为 AI 回复
    chat_history.append((user_input, ai_response))
    return chat_history

def main():
    with gr.Blocks() as demo:
        gr.Markdown("## AI 聊天界面")

        # 用于显示对话的组件
        chat_box = gr.Chatbot(label="对话")

        # 用于输入的文本框
        user_input = gr.Textbox(placeholder="输入你的消息...", label="输入框")

        # 用于更新对话的按钮
        send_button = gr.Button("发送")

        # 定义按钮点击事件
        send_button.click(
            fn=chat_with_ai,
            inputs=[user_input, chat_box],
            outputs=[chat_box]
        )
    demo.launch()

if __name__ == "__main__":
    main()