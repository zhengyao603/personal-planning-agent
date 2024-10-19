from typing import List

from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field

from database.mysql.mysql_config import get_mysql_connection, sql_execute

class ScheduleArgs(BaseModel):
    date: str = Field(description="""Date of schedule, need to be in "yyyy-mm-dd" format, Example: "2024-10-01" """)
    desc: str = Field(description="""Description of schedule, need to be at most 255 characters, Example: "play with my friend Julia" """)

def schedule_insert(date: str, desc: str) -> None:
    """
    Insert new schedule record.

    Args:
        date: Date of new schedule
        desc: Description of new schedule
    """

    insert_sql = f"""INSERT INTO t_schedule(date, description) VALUES ("{date}", "{desc}");"""
    conn = get_mysql_connection("localhost", 3306, "user", "123456", "db_agent")
    sql_execute(conn, insert_sql)

def schedule_delete(date: str) -> None:
    """
    Delete existing schedule record.

    Args:
        date: Date of existing schedule
    """

    # TODO: maybe optimize the logic to index db with both date and description (requires vector db and similarity??)
    delete_sql = f"""DELETE FROM t_schedule WHERE date="{date}";"""
    conn = get_mysql_connection("localhost", 3306, "user", "123456", "db_agent")
    sql_execute(conn, delete_sql)

def schedule_update(date: str, desc: str) -> None:
    """
    Modify existing schedule record.

    Args:
        date: Date of existing schedule
        desc: Description of modified schedule
    """

    update_sql = f"""UPDATE t_schedule SET description='{desc}' WHERE date="{date}";"""
    conn = get_mysql_connection("localhost", 3306, "user", "123456", "db_agent")
    sql_execute(conn, update_sql)

def get_mysql_toolkit() -> List[StructuredTool]:
    schedule_insert_tool = StructuredTool.from_function(
        func=schedule_insert,
        name="schedule_creation",
        description="""Useful for when you need to help user create a new schedule.
        Input of this tool are two separate strings: 1. date of the schedule, 2. description of the schedule.
        Example input: "2024-10-01", "play with my friend Julia".
        Note that the date parameter need to be in format of "yyyy-mm-dd", and description parameter need to be at most 255 characters.
        Note that the above example is just for demo purpose, you need to extract and summarize 'date' and 'description' parameters from user's input by yourself.""",
        args_schema=ScheduleArgs,
    )

    schedule_delete_tool = StructuredTool.from_function(
        func=schedule_delete,
        name="schedule_deletion",
        description="""Useful for when you need to help user delete an existing schedule.
        Input of this tool is one string: 1. date of the schedule.
        Example input: "2024-10-01".
        Note that the date parameter need to be in format of "yyyy-mm-dd".
        Note that the above example is just an example, you need to extract and summarize 'date' parameter from user's input by yourself.""",
    )

    schedule_update_tool = StructuredTool.from_function(
        func=schedule_update,
        name="schedule_modification",
        description="""Useful for when you need to help user modify an existing schedule.
        Input of this tool are two separate strings: 1. date of the schedule, 2. description of the schedule.
        Example input: "2024-10-01", "play with my friend Julia".
        Note that the date parameter need to be in format of "yyyy-mm-dd", and description parameter need to be at most 255 characters.
        Note that the above example is just for demo purpose, you need to extract and summarize 'date' and 'description' parameters from user's input by yourself.""",
        args_schema=ScheduleArgs,
    )

    return [schedule_insert_tool, schedule_delete_tool, schedule_update_tool]