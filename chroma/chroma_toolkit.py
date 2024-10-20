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