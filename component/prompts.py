from langchain import hub
from langchain_core.prompts import PromptTemplate

openai_tools_agent_prompt_template = hub.pull("hwchase17/openai-tools-agent")

sql_route_prompt_template = PromptTemplate.from_template(
    """Your are a perfect MySQL assistant. Given the user question below, classify it as either being about `DML(Data Manipulation Language)` or `DQL(Data Query Language)`.
    DQL is for only QUERYING database(i.e. SELECT), DML is for MANIPULATING database(i.e. INSERT/UPDATE/DELETE)
    Do not respond any word other than `DML` or `DQL`.
    Do not respond with more than one word.

<question>
{input}
</question>

Classification:"""
)

vector_route_prompt_template = PromptTemplate.from_template(
    """Your are a perfect Chroma assistant. Given the user question below, classify it as either being about `retrieval(Retrieve relevant document from Chroma)` or `manipulate(Create/Delete/Edit document in Chroma)`.
    `retrival` is for only QUERYING database(i.e. if user ask you to help him query some data), `manipulate` is for MANIPULATING database(i.e. if user ask you to help him create, deleter or edit some data).
    Do not respond any word other than `retrieval` or `manipulate`.
    Do not respond with more than one word.

<question>
{input}
</question>

Classification:"""
)

task_route_prompt_template = PromptTemplate.from_template(
    """Your are a perfect logical routing expert. Given the user question below, classify it as either being about `schedule` or `memo`.
    `schedule` is for user to record a schedule record, for example, "I need to record a schedule appointment on 2024-10-01 for an interview."
    `memo` is for user to record some notes or reflections, for example, "I want to create a new memo about my understanding of Langchain."
    Do not respond any word other than `schedule` or `memo`.
    Do not respond with more than one word.

<question>
{input}
</question>

Classification:"""
)


default_template = PromptTemplate.from_template(
    """Please just output 'Sorry, I don't know what you meant.'
    Not output any other words. Just output the sentence 'Sorry, I don't know what you meant.' """
)