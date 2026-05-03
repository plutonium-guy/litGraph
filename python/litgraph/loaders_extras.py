"""Optional document-loader adapters that wrap third-party Python libs.

Native loaders (`litgraph.loaders`) cover ~25 sources via Rust. This
module is the escape hatch for sources whose Python ecosystem is
mature enough that re-implementing them in Rust isn't worth it:
IMAP (stdlib), YouTube transcripts, Reddit (PRAW), Airtable.

Each adapter returns the same `Document`-shape dict (keys:
`page_content`, `metadata`) used by the rest of the pipeline so they
slot into splitters / vector stores without translation.

Example:

    from litgraph.loaders_extras import ImapLoader

    docs = ImapLoader(
        host="imap.gmail.com",
        username="user@example.com",
        password=os.environ["IMAP_PASSWORD"],
        mailbox="INBOX",
        max_messages=200,
    ).load()
    print(f"loaded {len(docs)} messages")
"""
from __future__ import annotations

import os
from typing import Any, Iterable


__all__ = [
    "ImapLoader",
    "YouTubeTranscriptLoader",
    "RedditLoader",
    "AirtableLoader",
]


def _doc(content: str, **metadata: Any) -> dict[str, Any]:
    """Build a `Document`-shape dict matching `litgraph.loaders` output."""
    return {"page_content": content, "metadata": dict(metadata)}


class ImapLoader:
    """Pull emails from any IMAP server via the Python stdlib
    (`imaplib`). No third-party dep. Defaults to fetching plain-text
    body; falls back to HTML stripped of tags.

    Args:
        host: IMAP server hostname (e.g. "imap.gmail.com").
        username: account login.
        password: account password / app-password.
        mailbox: folder to fetch (default "INBOX").
        max_messages: cap. Newest first.
        port: server port (default 993, SSL).
        use_ssl: use SSL transport (default True; required for Gmail).
        search_criteria: IMAP4 search string ("ALL", "UNSEEN",
            "SINCE 01-Jan-2025"). See RFC 3501 §6.4.4.
    """

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
    ) -> None:
        if max_messages <= 0:
            raise ValueError("max_messages must be positive")
        self.host = host
        self.username = username
        self.password = password
        self.mailbox = mailbox
        self.max_messages = max_messages
        self.port = port
        self.use_ssl = use_ssl
        self.search_criteria = search_criteria

    def load(self) -> list[dict[str, Any]]:
        import email
        import imaplib

        cls = imaplib.IMAP4_SSL if self.use_ssl else imaplib.IMAP4
        with cls(self.host, self.port) as conn:
            conn.login(self.username, self.password)
            conn.select(self.mailbox)
            status, data = conn.search(None, self.search_criteria)
            if status != "OK":
                raise RuntimeError(f"IMAP search failed: {status}")
            ids = data[0].split()
            # Newest first; cap.
            ids = list(reversed(ids))[: self.max_messages]
            docs: list[dict[str, Any]] = []
            for mid in ids:
                status, msg_data = conn.fetch(mid, "(RFC822)")
                if status != "OK" or not msg_data or msg_data[0] is None:
                    continue
                raw = msg_data[0][1] if isinstance(msg_data[0], tuple) else msg_data[0]
                msg = email.message_from_bytes(raw)
                body = _extract_email_body(msg)
                docs.append(_doc(
                    body,
                    source="imap",
                    mailbox=self.mailbox,
                    message_id=msg.get("Message-ID", ""),
                    subject=msg.get("Subject", ""),
                    sender=msg.get("From", ""),
                    date=msg.get("Date", ""),
                ))
            return docs


def _extract_email_body(msg: Any) -> str:
    """Return the plain-text body of an `email.message.Message`.
    Multipart messages: prefer `text/plain`, fallback to `text/html`
    stripped via a tiny regex (good enough for cold-store text)."""
    import re

    plain: str | None = None
    html: str | None = None
    if msg.is_multipart():
        for part in msg.walk():
            if part.is_multipart():
                continue
            ctype = part.get_content_type()
            try:
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes):
                    text = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
                else:
                    text = str(payload or "")
            except (LookupError, UnicodeDecodeError):
                continue
            if ctype == "text/plain" and plain is None:
                plain = text
            elif ctype == "text/html" and html is None:
                html = text
    else:
        payload = msg.get_payload(decode=True)
        if isinstance(payload, bytes):
            text = payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
        else:
            text = str(payload or "")
        if msg.get_content_type() == "text/html":
            html = text
        else:
            plain = text
    if plain:
        return plain
    if html:
        return re.sub(r"<[^>]+>", " ", html)
    return ""


class YouTubeTranscriptLoader:
    """Fetch the auto-generated transcript for a YouTube video.

    Lazy-imports `youtube_transcript_api`; install with
    `pip install youtube-transcript-api`.

    Args:
        video_ids: list of YouTube video IDs (the part after `v=` in
            a watch URL).
        languages: preferred languages, in priority order. Defaults
            to ["en"].
    """

    def __init__(self, video_ids: Iterable[str], languages: Iterable[str] = ("en",)) -> None:
        self.video_ids = list(video_ids)
        self.languages = list(languages)
        if not self.video_ids:
            raise ValueError("video_ids must not be empty")

    def load(self) -> list[dict[str, Any]]:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "youtube-transcript-api not installed. "
                "Run `pip install youtube-transcript-api` to use this loader."
            ) from e

        docs: list[dict[str, Any]] = []
        for vid in self.video_ids:
            try:
                segments = YouTubeTranscriptApi.get_transcript(vid, languages=self.languages)
            except Exception as e:
                # Don't abort the whole batch on one bad video; record
                # the error as metadata.
                docs.append(_doc(
                    "",
                    source="youtube",
                    video_id=vid,
                    error=str(e),
                ))
                continue
            text = "\n".join(s["text"] for s in segments if s.get("text"))
            docs.append(_doc(
                text,
                source="youtube",
                video_id=vid,
                segment_count=len(segments),
            ))
        return docs


class RedditLoader:
    """Pull posts (or comments) from a subreddit via PRAW.

    Lazy-imports `praw`; install with `pip install praw`. Reddit API
    requires an OAuth app — see PRAW docs for the credential flow.

    Args:
        subreddit: subreddit name without the `r/` prefix.
        client_id, client_secret, user_agent: PRAW credentials.
            `user_agent` is mandatory per Reddit's API terms.
        sort: "new", "hot", "top", "rising".
        limit: max posts.
        load_comments: also fetch top-level comments per post.
    """

    def __init__(
        self,
        subreddit: str,
        client_id: str | None = None,
        client_secret: str | None = None,
        user_agent: str | None = None,
        sort: str = "new",
        limit: int = 50,
        load_comments: bool = False,
    ) -> None:
        if sort not in ("new", "hot", "top", "rising"):
            raise ValueError(f"sort must be one of new/hot/top/rising, got {sort!r}")
        self.subreddit = subreddit.lstrip("/").removeprefix("r/")
        self.client_id = client_id or os.environ.get("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("REDDIT_CLIENT_SECRET")
        self.user_agent = user_agent or os.environ.get(
            "REDDIT_USER_AGENT", "litgraph-loader/0.1"
        )
        self.sort = sort
        self.limit = limit
        self.load_comments = load_comments

    def load(self) -> list[dict[str, Any]]:
        try:
            import praw  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "praw not installed. Run `pip install praw` to use this loader."
            ) from e

        reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent,
        )
        sub = reddit.subreddit(self.subreddit)
        listing = getattr(sub, self.sort)(limit=self.limit)
        docs: list[dict[str, Any]] = []
        for post in listing:
            body = post.selftext or post.title
            docs.append(_doc(
                body,
                source="reddit",
                subreddit=self.subreddit,
                post_id=post.id,
                title=post.title,
                author=str(post.author) if post.author else "",
                url=post.url,
                score=post.score,
                created_utc=post.created_utc,
            ))
            if self.load_comments:
                try:
                    post.comments.replace_more(limit=0)
                    for comment in post.comments[:20]:
                        docs.append(_doc(
                            comment.body or "",
                            source="reddit",
                            subreddit=self.subreddit,
                            post_id=post.id,
                            comment_id=comment.id,
                            author=str(comment.author) if comment.author else "",
                            score=comment.score,
                        ))
                except Exception:
                    # Skip comments on error — never abort the batch.
                    pass
        return docs


class AirtableLoader:
    """Pull rows from an Airtable base/table via the official
    `pyairtable` client.

    Lazy-imports `pyairtable`; install with `pip install pyairtable`.

    Args:
        base_id: Airtable base id (`appXXXXXXXXXXXXXX`).
        table_name: table name within the base.
        api_key: PAT or legacy API key. Falls back to
            `AIRTABLE_API_KEY` env.
        formula: Airtable formula filter (optional).
        view: view name (optional).
        max_records: cap.
    """

    def __init__(
        self,
        base_id: str,
        table_name: str,
        api_key: str | None = None,
        formula: str | None = None,
        view: str | None = None,
        max_records: int = 1000,
    ) -> None:
        self.base_id = base_id
        self.table_name = table_name
        self.api_key = api_key or os.environ.get("AIRTABLE_API_KEY")
        self.formula = formula
        self.view = view
        self.max_records = max_records

    def load(self) -> list[dict[str, Any]]:
        try:
            from pyairtable import Api  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "pyairtable not installed. "
                "Run `pip install pyairtable` to use this loader."
            ) from e
        if not self.api_key:
            raise ValueError("Airtable API key required (env AIRTABLE_API_KEY)")
        api = Api(self.api_key)
        table = api.table(self.base_id, self.table_name)
        kwargs: dict[str, Any] = {"max_records": self.max_records}
        if self.formula:
            kwargs["formula"] = self.formula
        if self.view:
            kwargs["view"] = self.view
        rows = table.all(**kwargs)
        docs: list[dict[str, Any]] = []
        for row in rows:
            fields = row.get("fields", {})
            # Concatenate all string fields as the page_content.
            content = "\n".join(
                f"{k}: {v}" for k, v in fields.items() if isinstance(v, (str, int, float))
            )
            docs.append(_doc(
                content,
                source="airtable",
                base_id=self.base_id,
                table=self.table_name,
                record_id=row.get("id", ""),
                created_time=row.get("createdTime", ""),
            ))
        return docs
