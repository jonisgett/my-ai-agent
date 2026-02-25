"""
Configuration loader — reads .env and provides typed settings.

Instead of scattering os.getenv() calls throughout the codebase,
we load everything once into a clean dataclass.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

@dataclass
class Config:
    """All agent configuration in one place."""

    # Core
    anthropic_api_key: str = ""
    model: str = "claude-sonnet-4-5-20250929"

    # Paths
    memory_dir: Path = Path("./memory")
    skills_dir: Path = Path("./skills")
    database_path: Path = Path("./memory/agent.db")

    # Heartbeat
    heartbeat_interval_minutes: int = 30

    # Gmail (optional)
    gmail_credentials_path: Path | None = None
    gmail_token_path: Path | None = None

    # Google Calendar (optional)
    gcal_credentials_path: Path | None = None
    gcal_token_path: Path | None = None

    # Slack (optional)
    slack_bot_token: str = ""
    slack_app_token: str = ""

    # Safety
    auto_approve_actions: list[str] = field(default_factory=lambda: [
        "read_file", "search_memory", "list_skills"
    ])
    skill_timeout_seconds: int = 30

def load_config(env_path: str | None = None) -> Config:
    """Load configuration from a .env file."""
    # Load the .env file into environment variables
    load_dotenv(env_path or ".env")

    config = Config(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        model=os.getenv("AGENT_MODEL", "claude-sonnet-4-5-20250929"),
        memory_dir=Path(os.getenv("MEMORY_DIR", "./memory")),
        skills_dir=Path(os.getenv("SKILLS_DIR", "./skills")),
        database_path=Path(os.getenv("DATABASE_PATH", "./memory/agent.db")),
        heartbeat_interval_minutes=int(os.getenv("HEARTBEAT_INTERVAL_MINUTES", "30")),
        slack_bot_token=os.getenv("SLACK_BOT_TOKEN", ""),
        slack_app_token=os.getenv("SLACK_APP_TOKEN", ""),
        skill_timeout_seconds=int(os.getenv("SKILL_TIMEOUT_SECONDS", "30")),
    )

    # Parse the comma-separated auto-approve list
    auto_approve = os.getenv("AUTO_APPROVE_ACTIONS", "read_file,search_memory,list_skills")
    config.auto_approve_actions = [
        action.strip()
        for action in auto_approve.split(",")
        if action.strip()
    ]

    # Optional paths (only set if the env var exists)
    gmail_creds = os.getenv("GMAIL_CREDENTIALS_PATH")
    if gmail_creds:
        config.gmail_credentials_path = Path(gmail_creds)
        config.gmail_token_path = Path(
            os.getenv("GMAIL_TOKEN_PATH", "./credentials/gmail_token.json")
        )

    return config