from typing import Any, Iterable, Optional

def create_deep_agent(
    model: Any,
    tools: Optional[Iterable[Any]] = None,
    system_prompt: Optional[str] = None,
    agents_md_path: Optional[str] = None,
    skills_dir: Optional[str] = None,
    max_iterations: int = 15,
    with_planning: bool = True,
    with_vfs: bool = True,
) -> Any: ...
