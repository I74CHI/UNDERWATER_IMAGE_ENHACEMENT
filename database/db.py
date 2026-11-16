"""
db.py — Thin database connection helper.

Reads credentials from environment variables (set in backend/.env).
Supports MSSQL (primary) with a MySQL fallback flag.

Environment variables required:
    DB_DRIVER  : "mssql" | "mysql"
    DB_HOST    : server host, e.g. "localhost"
    DB_PORT    : port number as string, e.g. "1433" (MSSQL) / "3306" (MySQL)
    DB_NAME    : database name, e.g. "aquaenhance"
    DB_USER    : login username
    DB_PASSWORD: login password
"""

import os


def get_connection():
    """Return a live DB-API 2.0 connection based on DB_DRIVER env var."""
    driver = os.environ.get("DB_DRIVER", "mssql").lower()

    host     = os.environ["DB_HOST"]
    port     = os.environ["DB_PORT"]
    name     = os.environ["DB_NAME"]
    user     = os.environ["DB_USER"]
    password = os.environ["DB_PASSWORD"]

    if driver == "mssql":
        import pyodbc  # pip install pyodbc
        # Use SERVER=host\instance if host contains a backslash (named instance
        # e.g. localhost\SQLEXPRESS). Named instances ignore port; default
        # instances use SERVER=host,port.
        if "\\" in host:
            server_token = host          # e.g. localhost\SQLEXPRESS
        else:
            server_token = f"{host},{port}"  # e.g. 192.168.1.5,1433
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={server_token};"
            f"DATABASE={name};"
            f"UID={user};"
            f"PWD={password};"
            "TrustServerCertificate=yes;"
        )
        return pyodbc.connect(conn_str)

    if driver == "mysql":
        import mysql.connector  # pip install mysql-connector-python
        return mysql.connector.connect(
            host=host, port=int(port), database=name,
            user=user, password=password
        )

    raise ValueError(f"Unsupported DB_DRIVER: '{driver}'. Use 'mssql' or 'mysql'.")
