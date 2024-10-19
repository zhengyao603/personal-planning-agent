import pymysql

from langchain_community.utilities import SQLDatabase

def get_mysql_db() -> SQLDatabase:
    mysql_host = "localhost:3306"
    mysql_user = "user"
    mysql_password = "123456"
    mysql_db = SQLDatabase.from_uri(f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}/db_agent")
    return mysql_db

def get_mysql_connection(host: str, port: int, user: str, password: str, database: str) -> pymysql.Connection:
    connection = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database
    )
    return connection

def sql_execute(conn, sql) -> None:
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql)
        conn.commit()
    except Exception as e:
        conn.rollback()
    finally:
        conn.close()