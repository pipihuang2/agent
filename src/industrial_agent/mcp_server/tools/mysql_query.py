"""MySQL database query tools for MCP server.

This module provides tools for querying data from MySQL databases.
"""

from typing import Any, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from industrial_agent.config import get_settings


class MySQLQueryManager:
    """Manager for MySQL database queries."""

    def __init__(self, connection_url: Optional[str] = None):
        """
        Initialize the MySQL query manager.

        Args:
            connection_url: SQLAlchemy connection URL. If not provided,
                           uses settings from environment.
        """
        if connection_url:
            self._connection_url = connection_url
        else:
            settings = get_settings()
            self._connection_url = settings.get_mysql_url()

        self._engine: Optional[Engine] = None

    def _get_engine(self) -> Engine:
        """Get or create SQLAlchemy engine."""
        if self._engine is None:
            self._engine = create_engine(self._connection_url)
        return self._engine

    def test_connection(self) -> dict[str, Any]:
        """
        Test the database connection.

        Returns:
            Dictionary with connection status and info.
        """
        try:
            engine = self._get_engine()
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            return {
                "success": True,
                "message": "Database connection successful",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def list_tables(self) -> dict[str, Any]:
        """
        List all tables in the database.

        Returns:
            Dictionary with list of table names.
        """
        try:
            engine = self._get_engine()
            with engine.connect() as conn:
                result = conn.execute(text("SHOW TABLES"))
                tables = [row[0] for row in result.fetchall()]
            return {
                "success": True,
                "tables": tables,
                "count": len(tables),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def describe_table(self, table_name: str) -> dict[str, Any]:
        """
        Get the schema/structure of a table.

        Args:
            table_name: Name of the table to describe.

        Returns:
            Dictionary with column information.
        """
        try:
            engine = self._get_engine()
            with engine.connect() as conn:
                result = conn.execute(text(f"DESCRIBE `{table_name}`"))
                columns = []
                for row in result.fetchall():
                    columns.append({
                        "name": row[0],
                        "type": row[1],
                        "null": row[2],
                        "key": row[3],
                        "default": row[4],
                        "extra": row[5],
                    })
            return {
                "success": True,
                "table": table_name,
                "columns": columns,
                "column_count": len(columns),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def execute_query(
        self,
        query: str,
        params: Optional[dict[str, Any]] = None,
        limit: int = 1000,
    ) -> dict[str, Any]:
        """
        Execute a SELECT query and return results.

        Args:
            query: SQL SELECT query to execute.
            params: Optional query parameters for parameterized queries.
            limit: Maximum number of rows to return (default 1000).

        Returns:
            Dictionary with query results.
        """
        # Security: Only allow SELECT queries
        query_upper = query.strip().upper()
        if not query_upper.startswith("SELECT"):
            return {
                "success": False,
                "error": "Only SELECT queries are allowed for security reasons",
            }

        # Add LIMIT if not present
        if "LIMIT" not in query_upper:
            query = f"{query.rstrip(';')} LIMIT {limit}"

        try:
            engine = self._get_engine()
            df = pd.read_sql(text(query), engine, params=params)

            # Convert to records format
            records = df.to_dict(orient="records")

            return {
                "success": True,
                "data": records,
                "row_count": len(records),
                "columns": list(df.columns),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def query_table(
        self,
        table_name: str,
        columns: Optional[list[str]] = None,
        where: Optional[str] = None,
        order_by: Optional[str] = None,
        limit: int = 1000,
    ) -> dict[str, Any]:
        """
        Query a table with optional filtering and sorting.

        Args:
            table_name: Name of the table to query.
            columns: List of columns to select (None for all columns).
            where: WHERE clause conditions (without 'WHERE' keyword).
            order_by: ORDER BY clause (without 'ORDER BY' keyword).
            limit: Maximum number of rows to return.

        Returns:
            Dictionary with query results.
        """
        # Build query
        cols = ", ".join(f"`{c}`" for c in columns) if columns else "*"
        query = f"SELECT {cols} FROM `{table_name}`"

        if where:
            query += f" WHERE {where}"
        if order_by:
            query += f" ORDER BY {order_by}"
        query += f" LIMIT {limit}"

        try:
            engine = self._get_engine()
            df = pd.read_sql(text(query), engine)
            records = df.to_dict(orient="records")

            return {
                "success": True,
                "table": table_name,
                "data": records,
                "row_count": len(records),
                "columns": list(df.columns),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def get_table_stats(self, table_name: str, column: str) -> dict[str, Any]:
        """
        Get statistical summary for a numeric column.

        Args:
            table_name: Name of the table.
            column: Name of the numeric column to analyze.

        Returns:
            Dictionary with statistical summary.
        """
        query = f"""
        SELECT
            COUNT(`{column}`) as count,
            AVG(`{column}`) as mean,
            MIN(`{column}`) as min,
            MAX(`{column}`) as max,
            STDDEV(`{column}`) as std
        FROM `{table_name}`
        WHERE `{column}` IS NOT NULL
        """

        try:
            engine = self._get_engine()
            df = pd.read_sql(text(query), engine)
            stats = df.iloc[0].to_dict()

            # Convert numpy types to Python types
            stats = {k: float(v) if v is not None else None for k, v in stats.items()}

            return {
                "success": True,
                "table": table_name,
                "column": column,
                "statistics": stats,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def get_column_values(
        self,
        table_name: str,
        column: str,
        limit: int = 10000,
    ) -> dict[str, Any]:
        """
        Get all values from a specific column (for analysis tools).

        Args:
            table_name: Name of the table.
            column: Name of the column.
            limit: Maximum number of values to return.

        Returns:
            Dictionary with column values as a list.
        """
        query = f"SELECT `{column}` FROM `{table_name}` WHERE `{column}` IS NOT NULL LIMIT {limit}"

        try:
            engine = self._get_engine()
            df = pd.read_sql(text(query), engine)
            values = df[column].tolist()

            return {
                "success": True,
                "table": table_name,
                "column": column,
                "values": values,
                "count": len(values),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }


# Global MySQL query manager instance
_mysql_manager: Optional[MySQLQueryManager] = None


def get_mysql_manager() -> MySQLQueryManager:
    """Get or create the global MySQL query manager instance."""
    global _mysql_manager
    if _mysql_manager is None:
        _mysql_manager = MySQLQueryManager()
    return _mysql_manager
