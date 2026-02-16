"""
Identity loader — reads the markdown files that define who the agent is.

This module reads SOUL.md, USER.md, and MEMORY.md and combines them
into a single string that gets included in every conversation with Claude.
"""

from pathlib import Path

IDENTITY_FILES = [
    ("SOUL.md", "Agent Identity"),
    ("USER.md", "User Profile"),
    ("MEMORY.md", "Decisions & Lessons"),
]

def load_identity(memory_dir: Path) -> str:
    """
    Load all identity markdown files and combine them into one string.

    This string gets included in the system prompt - it's what Claude reads
    at the start of every conversation to know who it is and who you are.

    Parameters
    ----------
    memory_dir : Path
        The directory containing SOUL.md, USER.md, MEMORY.md

    Returns
    -------
    str
        Combined content of all identity files
    """

    memory_dir = Path(memory_dir)
    sections = []

    for filename, label in IDENTITY_FILES:
        filepath = memory_dir / filename
        if filepath.exists():
            content = filepath.read_text(encoding="utf-8").strip()
            if content:
                sections.append(f"## {label}\n{content}")
    else:
        # If the file doesn't exist, add a helpful placeholder
        sections.append(
            f"## {label}\n"
            f"*(No {filename} found - create one to personalize your agent)*"
        )
    return "\n\n".join(sections)



