from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.chains.base import Chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnableSerializable
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent

from prompts import openai_tools_agent_prompt_template, sql_route_prompt_template
from database.mysql.mysql_config import get_mysql_db
from database.mysql.mysql_toolkit import get_mysql_toolkit

from langchain.globals import set_verbose, set_debug

class SchedulePlanningAgent:
    def __init__(self):
        self._model = None
        self._sql_dql_agent = None
        self._sql_dml_agent = None

    @property
    def model(self) -> ChatOpenAI:
        if self._model is None:
            self._model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        return self._model

    @property
    def sql_dql_agent(self) -> AgentExecutor:
        if self._sql_dql_agent is None:
            try:
                sql_db = get_mysql_db()
                sql_dql_toolkit = SQLDatabaseToolkit(db=sql_db, llm=self.model)
                self._sql_dql_agent = create_sql_agent(llm=self.model, toolkit=sql_dql_toolkit, verbose=True)
            except Exception as e:
                print(f"Error creating SQL DQL agent: {e}")
                raise
        return self._sql_dql_agent

    @property
    def sql_dml_agent(self) -> AgentExecutor:
        if self._sql_dml_agent is None:
            try:
                sql_dml_toolkit = get_mysql_toolkit()
                sql_dml_agent = create_tool_calling_agent(llm=self.model, prompt=openai_tools_agent_prompt_template, tools=sql_dml_toolkit)
                self._sql_dml_agent = AgentExecutor.from_agent_and_tools(agent=sql_dml_agent, tools=sql_dml_toolkit, verbose=True)
            except Exception as e:
                print(f"Error creating SQL DML agent: {e}")
                raise
        return self._sql_dml_agent

    @property
    def sql_rout_chain(self) -> Chain:
        return sql_route_prompt_template | self.model | StrOutputParser()

    @property
    def sql_full_chain(self) -> RunnableSerializable:
        return RunnableParallel({"operation": self.sql_rout_chain, "input": lambda x: x["input"]}) | RunnableLambda(self.sql_route)

    def sql_route(self, info) -> Chain:
        if "DML" in info["operation"].upper():
            return self.sql_dml_agent
        elif "DQL" in info["operation"].upper():
            return self.sql_dql_agent
        else:
            # TODO: maybe implement an unique default chain
            return self.sql_dql_agent


if __name__ == '__main__':
    load_dotenv()

    # set_debug( True)
    agent = SchedulePlanningAgent()
    # agent.sql_full_chain.invoke({"input": "I will have an interview for Tencent backend engineer on Oct 25th 2024. Help me record this schedule."})
    # agent.sql_full_chain.invoke({"input": "I will cancel the schedule on Oct 25th 2024. Help me delete this schedule."})
    agent.sql_full_chain.invoke({"input": "Can you please help me check what is my schedule on Oct 25th 2024?"})
