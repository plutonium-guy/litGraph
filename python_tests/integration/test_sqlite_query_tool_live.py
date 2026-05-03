"""Live integration: `SqliteQueryTool` driven by DeepSeek `ReactAgent`.

Seeds a tiny SQLite DB, gives the agent a SELECT-only tool over an
allowlisted table, asks for a count, verifies the answer threads
through.
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration


def test_react_agent_runs_sqlite_query(deepseek_chat):
    from litgraph.agents import ReactAgent
    from litgraph.tools import SqliteQueryTool

    with tempfile.TemporaryDirectory() as root:
        db_path = Path(root) / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE cities (name TEXT, country TEXT)")
        conn.executemany(
            "INSERT INTO cities VALUES (?, ?)",
            [
                ("Paris", "France"),
                ("Berlin", "Germany"),
                ("Rome", "Italy"),
                ("Madrid", "Spain"),
            ],
        )
        conn.commit()
        conn.close()

        tool = SqliteQueryTool(
            db_path=str(db_path),
            allowed_tables=["cities"],
            read_only=True,
            max_rows=100,
        )
        agent = ReactAgent(
            deepseek_chat,
            [tool],
            system_prompt=(
                "You have a sqlite_query tool. The DB has table `cities(name, "
                "country)`. To answer the user, write a SELECT query and call "
                "the tool. Reply with just the answer."
            ),
            max_iterations=5,
        )
        state = agent.invoke(
            "How many rows are in the cities table? Use the sqlite_query tool."
        )
        msgs = state["messages"]
        final = msgs[-1]
        text = final.get("content", "") if isinstance(final, dict) else str(final)
        if isinstance(text, list):
            text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))
        assert "4" in (text or ""), f"agent answer wrong: {final!r}"
