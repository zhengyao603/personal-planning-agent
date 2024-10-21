import gradio as gr

from component.agent import SchedulePlanningAgent

def agent_chat(question: str) -> str:
    agent = SchedulePlanningAgent()
    return agent.full_chain.invoke({"input": question})["output"]

def main() -> None:
    app = gr.Interface(
        fn=agent_chat,
        inputs=["text"],
        outputs=["text"],
    )
    app.launch(share=True)

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    main()