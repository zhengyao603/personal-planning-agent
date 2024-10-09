import gradio as gr

def greet(input):
    return """
8:00 AM - 9:00 AM: Breakfast and Morning Routine
- Enjoy a healthy breakfast and prepare for the day.

9:00 AM - 10:30 AM: Work on Project A
- Focus on completing key tasks and deadlines.

10:30 AM - 10:45 AM: Break
- Stretch and grab a quick snack.
...
"""

demo = gr.Interface(
    fn=greet,
    inputs=["text"],
    outputs=["text"],
)

if __name__ == "__main__":
    demo.launch()