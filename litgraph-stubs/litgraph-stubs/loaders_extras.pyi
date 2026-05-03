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


class OutlookLoader:
    access_token: str | None
    folder: str
    max_messages: int
    select: str
    filter: str | None
    def __init__(
        self,
        access_token: str | None = ...,
        folder: str = "inbox",
        max_messages: int = 50,
        select: str = "subject,from,receivedDateTime,bodyPreview,body",
        filter: str | None = ...,
    ) -> None: ...
    def load(self) -> list[dict[str, Any]]: ...


class HuggingFaceDatasetsLoader:
    dataset_name: str
    split: str
    text_column: str
    max_records: int
    config_name: str | None
    streaming: bool
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        text_column: str = "text",
        max_records: int = 1000,
        config_name: str | None = ...,
        streaming: bool = True,
    ) -> None: ...
    def load(self) -> list[dict[str, Any]]: ...


class TwitterLoader:
    query: str | None
    user_id: str | None
    bearer_token: str | None
    max_results: int
    tweet_fields: str
    def __init__(
        self,
        query: str | None = ...,
        bearer_token: str | None = ...,
        max_results: int = 100,
        tweet_fields: str = "id,text,created_at,author_id,public_metrics",
        user_id: str | None = ...,
    ) -> None: ...
    def load(self) -> list[dict[str, Any]]: ...
