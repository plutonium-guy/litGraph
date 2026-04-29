from typing import Any

class TextLoader:
    def __init__(self, path: str) -> None: ...
    def load(self) -> list[dict[str, Any]]: ...

class JsonLinesLoader:
    def __init__(self, path: str, content_field: str = "content") -> None: ...
    def load(self) -> list[dict[str, Any]]: ...

class MarkdownLoader:
    def __init__(self, path: str) -> None: ...
    def load(self) -> list[dict[str, Any]]: ...

class DirectoryLoader:
    def __init__(self, root: str, glob: str = "**/*", follow_symlinks: bool = False) -> None: ...
    def load(self) -> list[dict[str, Any]]: ...

class WebLoader:
    def __init__(
        self,
        url: str,
        timeout_s: int = 30,
        user_agent: str | None = None,
    ) -> None: ...
    def load(self) -> list[dict[str, Any]]: ...

class CsvLoader:
    def __init__(
        self,
        path: str,
        content_column: str | None = None,
        delimiter: str = ",",
        max_rows: int | None = None,
    ) -> None: ...
    def load(self) -> list[dict[str, Any]]: ...

class HtmlLoader:
    def __init__(
        self,
        path: str | None = None,
        html: str | None = None,
        strip_boilerplate: bool = True,
    ) -> None: ...
    def load(self) -> list[dict[str, Any]]: ...

class JsonLoader:
    def __init__(
        self,
        path: str,
        pointer: str | None = None,
        content_field: str | None = None,
    ) -> None: ...
    def load(self) -> list[dict[str, Any]]: ...
