from typing import Any, Iterable, Optional, Sequence, Union

# Either a tuple/list of strings, or a single string (treated as a one-segment
# namespace).
Namespace = Union[Sequence[str], str]

class StoreItem(dict):
    namespace: list[str]
    key: str
    value: Any
    expires_at_ms: Optional[int]
    created_at_ms: int
    updated_at_ms: int

class InMemoryStore:
    def __init__(self) -> None: ...
    def put(
        self,
        namespace: Namespace,
        key: str,
        value: Any,
        ttl_ms: Optional[int] = None,
    ) -> None: ...
    def get(self, namespace: Namespace, key: str) -> Optional[StoreItem]: ...
    def delete(self, namespace: Namespace, key: str) -> bool: ...
    def pop(self, namespace: Namespace, key: str) -> StoreItem: ...
    def search(
        self,
        namespace_prefix: Namespace,
        query_text: Optional[str] = None,
        matches: Optional[Iterable[tuple[str, Any]]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> list[StoreItem]: ...
    def list_namespaces(
        self,
        prefix: Optional[Namespace] = None,
        limit: Optional[int] = None,
    ) -> list[list[str]]: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
