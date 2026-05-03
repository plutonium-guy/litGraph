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
from typing import Any, Iterable, Mapping


__all__ = [
    "ImapLoader",
    "YouTubeTranscriptLoader",
    "RedditLoader",
    "AirtableLoader",
    "OutlookLoader",
    "HuggingFaceDatasetsLoader",
    "TwitterLoader",
    "WhatsAppCloudLoader",
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


class OutlookLoader:
    """Pull emails from Outlook / Microsoft 365 via the Microsoft
    Graph API. Lazy-imports `requests`; install with
    `pip install requests`.

    Auth: uses an OAuth2 access token (passed in or read from
    `MS_GRAPH_TOKEN`). Acquire via the `msal` library or device-code
    flow — out of scope for this loader; the token is the only input.

    Args:
        access_token: Bearer token with `Mail.Read` (or
            `Mail.ReadBasic`) scope.
        folder: Outlook folder to fetch (default "inbox"). Use
            `mailFolders/{id}` for nested folders.
        max_messages: cap.
        select: comma-separated `$select` fields. Default covers the
            ones used in the document body.
        filter: optional Graph `$filter` clause (e.g.
            "receivedDateTime ge 2025-01-01T00:00:00Z").
    """

    def __init__(
        self,
        access_token: str | None = None,
        folder: str = "inbox",
        max_messages: int = 50,
        select: str = "subject,from,receivedDateTime,bodyPreview,body",
        filter: str | None = None,
    ) -> None:
        if max_messages <= 0:
            raise ValueError("max_messages must be positive")
        self.access_token = access_token or os.environ.get("MS_GRAPH_TOKEN")
        if not self.access_token:
            raise ValueError(
                "access_token required (env MS_GRAPH_TOKEN). "
                "Acquire via msal or the device-code flow."
            )
        self.folder = folder
        self.max_messages = max_messages
        self.select = select
        self.filter = filter

    def load(self) -> list[dict[str, Any]]:
        try:
            import requests  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError("requests not installed. Run `pip install requests`.") from e
        import re

        url = f"https://graph.microsoft.com/v1.0/me/{self.folder}/messages"
        params: dict[str, Any] = {
            "$top": min(self.max_messages, 100),
            "$select": self.select,
            "$orderby": "receivedDateTime desc",
        }
        if self.filter:
            params["$filter"] = self.filter
        headers = {"Authorization": f"Bearer {self.access_token}"}

        docs: list[dict[str, Any]] = []
        fetched = 0
        while url and fetched < self.max_messages:
            r = requests.get(url, headers=headers, params=params if "$select" in (params or {}) else None, timeout=30)
            r.raise_for_status()
            payload = r.json()
            for item in payload.get("value", []):
                if fetched >= self.max_messages:
                    break
                body = item.get("body") or {}
                content = body.get("content", "") or item.get("bodyPreview", "")
                # Strip HTML if returned as html.
                if body.get("contentType", "").lower() == "html":
                    content = re.sub(r"<[^>]+>", " ", content)
                sender = (item.get("from") or {}).get("emailAddress", {}).get("address", "")
                docs.append(_doc(
                    content,
                    source="outlook",
                    folder=self.folder,
                    message_id=item.get("id", ""),
                    subject=item.get("subject", ""),
                    sender=sender,
                    received=item.get("receivedDateTime", ""),
                ))
                fetched += 1
            url = payload.get("@odata.nextLink")
            params = {}  # nextLink already encodes the params
        return docs


class HuggingFaceDatasetsLoader:
    """Iterate a HuggingFace dataset and emit Documents. Lazy-imports
    `datasets`; install with `pip install datasets`.

    Args:
        dataset_name: the canonical HF id (e.g. "squad", "imdb").
        split: "train" / "validation" / "test".
        text_column: column whose value becomes `page_content`.
        max_records: cap.
        config_name: dataset config / subset (optional).
        streaming: use streaming mode (recommended for large datasets;
            doesn't materialise the full table in RAM).
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        text_column: str = "text",
        max_records: int = 1000,
        config_name: str | None = None,
        streaming: bool = True,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.text_column = text_column
        self.max_records = max_records
        self.config_name = config_name
        self.streaming = streaming

    def load(self) -> list[dict[str, Any]]:
        try:
            from datasets import load_dataset  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "datasets not installed. Run `pip install datasets`."
            ) from e
        ds = load_dataset(
            self.dataset_name,
            self.config_name,
            split=self.split,
            streaming=self.streaming,
        )
        docs: list[dict[str, Any]] = []
        for i, row in enumerate(ds):
            if i >= self.max_records:
                break
            text = row.get(self.text_column, "") or ""
            md = {k: v for k, v in row.items() if k != self.text_column and isinstance(v, (str, int, float, bool))}
            docs.append(_doc(
                str(text),
                source="huggingface_datasets",
                dataset=self.dataset_name,
                split=self.split,
                row_index=i,
                **md,
            ))
        return docs


class TwitterLoader:
    """Pull tweets via the X (Twitter) API v2. Lazy-imports `tweepy`;
    install with `pip install tweepy`. Requires a bearer token from
    a paid X developer account — Twitter retired free tiers in 2023.

    Args:
        bearer_token: X API v2 bearer token. Falls back to
            `TWITTER_BEARER_TOKEN` env.
        query: search query (X API v2 search syntax).
        max_results: cap (X enforces 10–100 per page; the loader
            paginates up to this total).
        tweet_fields: comma-separated `tweet.fields` parameter.
        user_id: alternative to `query` — fetch a specific user's
            timeline. Mutually exclusive with `query`.
    """

    def __init__(
        self,
        query: str | None = None,
        bearer_token: str | None = None,
        max_results: int = 100,
        tweet_fields: str = "id,text,created_at,author_id,public_metrics",
        user_id: str | None = None,
    ) -> None:
        if (query is None) == (user_id is None):
            raise ValueError("specify exactly one of `query` or `user_id`")
        if max_results <= 0:
            raise ValueError("max_results must be positive")
        self.query = query
        self.user_id = user_id
        self.bearer_token = bearer_token or os.environ.get("TWITTER_BEARER_TOKEN")
        if not self.bearer_token:
            raise ValueError("bearer_token required (env TWITTER_BEARER_TOKEN)")
        self.max_results = max_results
        self.tweet_fields = tweet_fields

    def load(self) -> list[dict[str, Any]]:
        try:
            import tweepy  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "tweepy not installed. Run `pip install tweepy`."
            ) from e
        client = tweepy.Client(bearer_token=self.bearer_token)
        per_page = min(self.max_results, 100)
        per_page = max(per_page, 10)  # X API minimum
        docs: list[dict[str, Any]] = []
        # tweepy.Paginator handles next-token cursors transparently.
        if self.query:
            paginator = tweepy.Paginator(
                client.search_recent_tweets,
                query=self.query,
                tweet_fields=self.tweet_fields,
                max_results=per_page,
                limit=(self.max_results + per_page - 1) // per_page,
            )
        else:
            paginator = tweepy.Paginator(
                client.get_users_tweets,
                id=self.user_id,
                tweet_fields=self.tweet_fields,
                max_results=per_page,
                limit=(self.max_results + per_page - 1) // per_page,
            )
        for response in paginator:
            for tweet in (response.data or []):
                if len(docs) >= self.max_results:
                    return docs
                metrics = getattr(tweet, "public_metrics", {}) or {}
                docs.append(_doc(
                    tweet.text or "",
                    source="twitter",
                    tweet_id=str(tweet.id),
                    author_id=str(getattr(tweet, "author_id", "")),
                    created_at=str(getattr(tweet, "created_at", "")),
                    retweets=metrics.get("retweet_count", 0),
                    likes=metrics.get("like_count", 0),
                ))
        return docs


class WhatsAppCloudLoader:
    """Pull conversation history from the WhatsApp Business Cloud API.

    The Cloud API exposes message webhooks (push) — there's no read
    endpoint for arbitrary chat history. This loader supports two
    workflows:

    1. **Webhook capture** (most common): your application stores
       inbound messages somewhere durable (Postgres, S3); pass the
       directory or URL via `messages_source` and this loader yields
       Documents.
    2. **Server-stored archives**: Meta's Cloud API exposes
       `/{phone_number_id}/messages` for *outbound* / template-driven
       messages and a media-fetch endpoint per message id. The
       loader's `phone_number_id` + `access_token` mode pulls *those*
       — see Meta's docs for the limits (no inbound history without
       webhooks).

    Args:
        access_token: Cloud API permanent or system-user token.
            Falls back to `WHATSAPP_CLOUD_TOKEN` env.
        phone_number_id: phone-number id from the WhatsApp Manager.
        messages_source: alternative — a list-of-dicts already
            captured from your webhook (for batch ingest).
        max_messages: cap.

    Raises `ImportError` lazily when `requests` isn't installed.
    """

    def __init__(
        self,
        access_token: str | None = None,
        phone_number_id: str | None = None,
        messages_source: Iterable[Mapping[str, Any]] | None = None,
        max_messages: int = 200,
    ) -> None:
        if max_messages <= 0:
            raise ValueError("max_messages must be positive")
        if messages_source is None and (access_token is None or phone_number_id is None):
            self.access_token = os.environ.get("WHATSAPP_CLOUD_TOKEN") or access_token
            if not self.access_token or not phone_number_id:
                raise ValueError(
                    "Provide either messages_source (webhook capture) "
                    "or both access_token (env WHATSAPP_CLOUD_TOKEN) + phone_number_id."
                )
        self.access_token = access_token or os.environ.get("WHATSAPP_CLOUD_TOKEN")
        self.phone_number_id = phone_number_id
        self.messages_source = list(messages_source) if messages_source is not None else None
        self.max_messages = max_messages

    def load(self) -> list[dict[str, Any]]:
        # Webhook-captured path — pure Python, no network.
        if self.messages_source is not None:
            return [
                _doc(
                    str(m.get("text", {}).get("body") if isinstance(m.get("text"), dict) else m.get("body", "")),
                    source="whatsapp_cloud",
                    message_id=m.get("id", ""),
                    sender=m.get("from", ""),
                    timestamp=m.get("timestamp", ""),
                )
                for m in self.messages_source[: self.max_messages]
                if m.get("type") in (None, "text")
            ]
        # Cloud-API path — pulls outbound message records (Cloud API
        # does NOT expose inbound chat history; capture via webhooks).
        try:
            import requests  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "requests not installed. Run `pip install requests`."
            ) from e
        url = f"https://graph.facebook.com/v20.0/{self.phone_number_id}/messages"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        r = requests.get(url, headers=headers, params={"limit": self.max_messages}, timeout=30)
        r.raise_for_status()
        payload = r.json()
        docs: list[dict[str, Any]] = []
        for item in payload.get("data", [])[: self.max_messages]:
            text = ""
            if isinstance(item.get("text"), dict):
                text = item["text"].get("body", "")
            elif isinstance(item.get("body"), str):
                text = item["body"]
            docs.append(_doc(
                text,
                source="whatsapp_cloud",
                message_id=item.get("id", ""),
                direction=item.get("direction", "outbound"),
                timestamp=item.get("timestamp", ""),
            ))
        return docs
