from typing import List
from pydantic import BaseModel, Field

from langchain_core.tools import StructuredTool
from langchain_core.documents import Document

from chroma.chroma_config import get_chroma_db

class MemoCreateArgs(BaseModel):
    title: str = Field(description="""Title of memo, no more than 50 characters, Example: "Application of Langchain" """)
    date: str = Field(description="""Date of memo, need to be in "yyyy-mm-dd" format, Example: "2024-10-01" """)
    content: str = Field(description="""Content of memo, no more than 255 characters, Example: "play with my friend Julia" """)

class MemoEditArgs(BaseModel):
    title: str = Field(description="""Title of memo, no more than 50 characters, Example: "Application of Langchain" """)
    content: str = Field(description="""Content of memo, no more than 255 characters, Example: "play with my friend Julia" """)

def memo_create(title:str, date: str, content: str) -> None:
    """
    Create a new memo record.

    Args:
        title: Title of new memo
        date: Date of new memo
        content: Content of new memo
    """
    chroma_db = get_chroma_db()
    documents = [
        Document(
            page_content=content,
            metadata={"date": date},
            id=title
        )
    ]
    chroma_db.add_documents(documents)

def memo_edit(title:str, content: str) -> None:
    """
    Edit an existing memo record.

    Args:
        title: Title of existing memo
        content: Content of existing memo
    """
    chroma_db = get_chroma_db()
    chroma_db.update_document(title, Document(page_content=content))

def memo_delete(title:str) -> None:
    """
    Edit an existing memo record.

    Args:
        title: Title of existing memo
    """
    chroma_db = get_chroma_db()
    chroma_db.delete(ids=[title])

def get_chroma_toolkit() -> List[StructuredTool]:
    memo_create_tool = StructuredTool.from_function(
        func=memo_create,
        name="memo_creation",
        description="""Useful for when you need to help user create a new memo.
            Input of this tool are three separate strings: 1. title of the memo, 2. date of the memo, 3. content of memo.
            Example input: "Have Fun Tomorrow!!", "2024-10-01", "play with my friend Julia".
            Note that the title parameter need to be at most 50 characters, date parameter need to be in format of "yyyy-mm-dd", and content parameter need to be at most 255 characters.
            Note that the above example is just for demo purpose, you need to understand user's input, extract and summarize 'title', 'date' and 'content' parameters by yourself.""",
        args_schema=MemoCreateArgs,
    )

    memo_edit_tool = StructuredTool.from_function(
        func=memo_edit,
        name="memo_edit",
        description="""Useful for when you need to help user edit an existing memo.
            Input of this tool are two separate strings: 1. title of the memo, 2. content of the memo.
            Example input: "Have Fun Tomorrow!!", "play with my friend Julia, remember to bring gift for her".
            Note that the title parameter need to be at most 50 characters and content parameter need to be at most 255 characters.
            Note that the above example is just for demo purpose, you need to understand user's input, extract and summarize 'title' and 'content' parameters by yourself.""",
    )

    memo_delete_tool = StructuredTool.from_function(
        func=memo_delete,
        name="memo_delete",
        description="""Useful for when you need to help user delete an existing memo.
                Input of this tool is only one string: 1. title of the memo.
                Example input: "Have Fun Tomorrow!!".
                Note that the title parameter need to be at most 50 characters.
                Note that the above example is just for demo purpose, you need to understand user's input, extract and summarize 'title' parameter by yourself.""",
    )

    return [memo_create_tool, memo_edit_tool, memo_delete_tool]

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    memo_create("Understanding Machine Learning Basics", "2023-04-15", """Today, I delved into the fundamentals of machine learning (ML), a subfield of artificial intelligence that focuses on building systems capable of learning from data and improving their performance over time without explicit programming. The core idea is to use algorithms to parse data, learn from it, and then make informed decisions based on the learned patterns.

Key concepts include supervised and unsupervised learning. Supervised learning involves training a model on a labeled dataset, meaning the output is known. Common algorithms in this category are linear regression and decision trees. Unsupervised learning, on the other hand, deals with unlabeled data, and the system tries to find hidden patterns or intrinsic structures. Clustering and association are popular techniques here.

Understanding the importance of data quality and preprocessing is crucial, as garbage in results in garbage out. Feature selection and extraction are vital steps that enhance model performance. Tools like Python's scikit-learn library are invaluable for implementing these algorithms.

Overall, machine learning is transforming industries by enabling data-driven decision-making, and mastering its basics is essential for future technological advancements.""")
    memo_create("Exploring Cloud Computing", "2023-05-10", """Today, I explored cloud computing, a paradigm that enables on-demand network access to a shared pool of configurable computing resources. This includes servers, storage, applications, and services that can be rapidly provisioned with minimal management effort.

Cloud computing is categorized into three main service models: Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and Software as a Service (SaaS). IaaS provides virtualized computing resources over the internet, PaaS offers a platform allowing customers to develop, run, and manage applications, and SaaS delivers software applications over the web.

The benefits of cloud computing are significant, including cost efficiency, scalability, flexibility, and improved collaboration. However, it also comes with challenges such as security concerns and potential downtime.

Major providers like Amazon Web Services (AWS), Microsoft Azure, and Google Cloud Platform offer a variety of services that cater to different organizational needs. As businesses increasingly shift to the cloud, understanding these services and their applications is critical for leveraging digital transformation.""")
    memo_create("Introduction to Golang Backend Development", "2023-06-25", """Today, I started learning about Golang (Go) for backend development, a statically typed, compiled language designed by Google. Golang is known for its simplicity, efficiency, and robust concurrency support, making it ideal for developing scalable web services and APIs.

Go's syntax is concise and easy to understand, which aids in writing clean and maintainable code. Its built-in concurrency model, powered by goroutines and channels, allows developers to efficiently handle multiple tasks simultaneously, a crucial feature for backend systems.

The language is also praised for its performance, comparable to C/C++, due to its compiled nature. Additionally, Golang has a strong standard library that simplifies tasks such as HTTP handling, JSON parsing, and database interaction.

I explored frameworks like Gin and Echo that provide additional functionality and streamline the development process. Golang's growing community and extensive documentation are excellent resources for new developers.

Overall, Golang presents a compelling option for backend development, balancing performance, simplicity, and scalability. As I continue to learn, I aim to build a small web service to apply these concepts practically.""")