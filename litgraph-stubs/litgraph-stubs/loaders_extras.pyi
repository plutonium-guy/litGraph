"""Optional document-loader adapters (lazy imports)."""
from __future__ import annotations
from typing import Any, Iterable


class ImapLoader:
    host: str
    username: str
    password: str
    mailbox: str
    max_messages: int
    port: int
    use_ssl: bool
    search_criteria: str
    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        mailbox: str = "INBOX",
        max_messages: int = 100,
        port: int = 993,
        use_ssl: bool = True,
        search_criteria: str = "ALL",
    ) -> None: ...
    def load(self) -> list[dict[str, Any]]: ...


class YouTubeTranscriptLoader:
    video_ids: list[str]
    languages: list[str]
    def __init__(self, video_ids: Iterable[str], languages: Iterable[str] = ...) -> None: ...
    def load(self) -> list[dict[str, Any]]: ...


class RedditLoader:
    subreddit: str
    client_id: str | None
    client_secret: str | None
    user_agent: str | None
    sort: str
    limit: int
    load_comments: bool
    def __init__(
        self,
        subreddit: str,
        client_id: str | None = ...,
        client_secret: str | None = ...,
        user_agent: str | None = ...,
        sort: str = "new",
        limit: int = 50,
        load_comments: bool = False,
    ) -> None: ...
    def load(self) -> list[dict[str, Any]]: ...


class AirtableLoader:
    base_id: str
    table_name: str
    api_key: str | None
    formula: str | None
    view: str | None
    max_records: int
    def __init__(
        self,
        base_id: str,
        table_name: str,
        api_key: str | None = ...,
        formula: str | None = ...,
        view: str | None = ...,
        max_records: int = 1000,
    ) -> None: ...
    def load(self) -> list[dict[str, Any]]: ...
