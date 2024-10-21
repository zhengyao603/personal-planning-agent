from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.chains.base import Chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnableSerializable
from langchain_core.tools import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent

from component.utils import load_config
from component.prompts import (openai_tools_agent_prompt_template, sql_route_prompt_template,
                     vector_route_prompt_template, task_route_prompt_template, default_template)

import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from mysql.mysql_config import get_mysql_db
from mysql.mysql_toolkit import get_mysql_toolkit
from chroma.chroma_config import get_chroma_db
from chroma.chroma_toolkit import get_chroma_toolkit

class SchedulePlanningAgent:
    def __init__(self):
        self._model = None
        self._sql_dql_agent = None
        self._sql_dml_agent = None
        self._vector_query_agent = None
        self._vector_manipulate_agent = None

    @property
    def model(self) -> ChatOpenAI:
        if self._model is None:
            self._model = ChatOpenAI(
                model=load_config()["llm"]["chat"],
                temperature=0
            )
        return self._model

    @property
    def sql_dql_agent(self) -> AgentExecutor:
        if self._sql_dql_agent is None:
            try:
                sql_db = get_mysql_db()
                sql_dql_toolkit = SQLDatabaseToolkit(
                    db=sql_db,
                    llm=self.model
                )
                self._sql_dql_agent = create_sql_agent(
                    llm=self.model,
                    toolkit=sql_dql_toolkit,
                    verbose=True
                )
            except Exception as e:
                print(f"Error creating SQL DQL agent: {e}")
                raise
        return self._sql_dql_agent

    @property
    def sql_dml_agent(self) -> AgentExecutor:
        if self._sql_dml_agent is None:
            try:
                sql_dml_toolkit = get_mysql_toolkit()
                sql_dml_agent = create_tool_calling_agent(
                    llm=self.model,
                    prompt=openai_tools_agent_prompt_template,
                    tools=sql_dml_toolkit
                )
                self._sql_dml_agent = AgentExecutor.from_agent_and_tools(
                    agent=sql_dml_agent,
                    tools=sql_dml_toolkit,
                    verbose=True
                )
            except Exception as e:
                print(f"Error creating SQL DML agent: {e}")
                raise
        return self._sql_dml_agent

    @property
    def vector_query_agent(self) -> AgentExecutor:
        if self._vector_query_agent is None:
            try:
                chroma_db = get_chroma_db()
                retriever = chroma_db.as_retriever()
                vector_query_toolkit = [create_retriever_tool(
                    retriever=retriever,
                    name="search_memo_record",
                    description="""Searches and returns memos from vector database.
                    Need to understand, extract and summarize memo title, memo date and memo content from users' input.
                    Use memo content for similarity search from vector database, and memo date for metadata filtering.
                    Note that date parameter need to be in format of "yyyy-mm-dd", you might need to parse it before using it for filtering. """
                )]
                vector_query_agent = create_tool_calling_agent(
                    llm=self.model,
                    tools=vector_query_toolkit,
                    prompt=openai_tools_agent_prompt_template,
                )
                self._vector_query_agent = AgentExecutor.from_agent_and_tools(
                    agent=vector_query_agent,
                    tools=vector_query_toolkit,
                    verbose=True
                )
            except Exception as e:
                print(f"Error creating VECTOR MANIPULATE agent: {e}")
                raise
        return self._vector_query_agent

    @property
    def vector_manipulate_agent(self) -> AgentExecutor:
        if self._vector_manipulate_agent is None:
            try:
                vector_manipulate_toolkit = get_chroma_toolkit()
                vector_manipulate_agent = create_tool_calling_agent(
                    llm=self.model,
                    prompt=openai_tools_agent_prompt_template,
                    tools=vector_manipulate_toolkit
                )
                self._vector_manipulate_agent = AgentExecutor.from_agent_and_tools(
                    agent=vector_manipulate_agent,
                    tools=vector_manipulate_toolkit,
                    verbose=True
                )
            except Exception as e:
                print(f"Error creating VECTOR MANIPULATE agent: {e}")
                raise
        return self._vector_manipulate_agent

    @property
    def sql_route_chain(self) -> Chain:
        return sql_route_prompt_template | self.model | StrOutputParser()

    @property
    def sql_full_chain(self) -> RunnableSerializable:
        return RunnableParallel({"operation": self.sql_route_chain, "input": lambda x: x["input"]}) | RunnableLambda(
            self.sql_route
        )

    @property
    def vector_route_chain(self) -> Chain:
        return vector_route_prompt_template | self.model | StrOutputParser()

    @property
    def vector_full_chain(self) -> RunnableSerializable:
        return RunnableParallel({"operation": self.vector_route_chain, "input": lambda x: x["input"]}) | RunnableLambda(
            self.vector_route
        )

    @property
    def task_route_chain(self) -> Chain:
        return task_route_prompt_template | self.model | StrOutputParser()

    @property
    def full_chain(self) -> RunnableSerializable:
        return RunnableParallel({"task": self.task_route_chain, "input": lambda x: x["input"]}) | RunnableLambda(
            self.task_route
        )

    @property
    def default_chain(self) -> Chain:
        return default_template | self.model | StrOutputParser()

    def sql_route(self, info) -> Chain:
        if "DML" in info["operation"].upper():
            return self.sql_dml_agent
        elif "DQL" in info["operation"].upper():
            return self.sql_dql_agent
        else:
            return self.default_chain

    def vector_route(self, info) -> Chain:
        if "retrieval" in info["operation"].lower():
            return self.vector_query_agent
        elif "manipulate" in info["operation"].lower():
            return self.vector_manipulate_agent
        else:
            return self.default_chain

    def task_route(self, info) -> RunnableSerializable:
        if "schedule" in info["task"].lower():
            return self.sql_full_chain
        elif "memo" in info["task"].lower():
            return self.vector_full_chain
        else:
            return self.default_chain