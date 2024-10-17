from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def division():
    # Load environment variables from .env
    load_dotenv()

    # Create a ChatOpenAI model
    model = ChatOpenAI(
        model="gpt-3.5-turbo-0125",
    )

    # Invoke the model with a message
    result = model.invoke("What is 81 divided by 9?")
    print("Full result:")
    print(result)
    print("Content only:")
    print(result.content)

if __name__ == "__main__":
    division()