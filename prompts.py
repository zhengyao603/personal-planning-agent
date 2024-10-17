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