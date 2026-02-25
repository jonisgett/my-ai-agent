"""
Skill Loader — discovers and loads SKILL.md files from the skills/ directory.

Drop a folder with a SKILL.md into skills/ and it's instantly available.
No public registry, no supply chain risk.
"""

import re
import yaml
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class Skill:
    """A loaded skill with its metadata and instructions."""
    name: str
    description: str
    version: str
    triggers: list[str]
    instructions: str
    path: Path
    files: list[str] = field(default_factory=list)

class SkillRegistry:
    """Discovers and manages local skills."""

    def __init__(self, skills_dir: Path):
        self.skills_dir = Path(skills_dir)
        self.registry: dict[str, Skill] = {}
        self._discover()

    def _discover(self):
        """Scan the skills directory for SKILL.md files."""
        if not self.skills_dir.exists():
            self.skills_dir.mkdir(parents=True, exist_ok=True)
            return

        for skill_dir in self.skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue

            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            try:
                skill = self._parse_skill(skill_md, skill_dir)
                self.registry[skill.name] = skill
            except Exception as e:
                print(f"⚠️  Failed to load skill from {skill_dir}: {e}")

        if not self.skills_dir.exists():
            self.skills_dir.mkdir(parents=True, exist_ok=True)
            return
        
        for skill_dir in self.skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            try:
                skill = self._parse_skill(skill_md, skill_dir)
                self.registry[skill.name] = skill
            except Exception as e:
                print(f"⚠️  Failed to load skill from {skill_dir}: {e}")

    def _parse_skill(self, skill_md: Path, skill_dir: Path) -> Skill:
        """Parse a SKILL.md file with YAML frontmatter."""
        content = skill_md.read_text(encoding="utf-8")

        # Extract YAML frontmatter (between --- markers)
        frontmatter = {}
        body = content

        fm_match = re.match(
            r"^---\s*\n(.*?)\n---\s*\n(.*)",
            content,
            re.DOTALL
        )

        if fm_match:
            try:
                frontmatter = yaml.safe_load(fm_match.group(1)) or {}
            except yaml.YAMLError:
                pass  # Bad YAML — use defaults
            body = fm_match.group(2)

        # List other files in the skill directory
        files = [
            str(f.relative_to(skill_dir))
            for f in skill_dir.rglob("*")
            if f.is_file() and f.name != "SKILL.md"
        ]

        return Skill(
            name=frontmatter.get("name", skill_dir.name),
            description=frontmatter.get("description", "(no description)"),
            version=frontmatter.get("version", "0.1"),
            triggers=frontmatter.get("triggers", []),
            instructions=body.strip(),
            path=skill_dir,
            files=files,
        )
    
    def get(self, name: str) -> Skill | None:
        """Get a skill by name."""
        return self.registry.get(name)

    def describe_all(self) -> str:
        """Tier 1: brief description of all skills."""
        if not self.registry:
            return "No skills loaded. Add folders to skills/ directory."

        lines = []
        for name, skill in sorted(self.registry.items()):
            lines.append(f"- **{name}**: {skill.description}")

        return "\n".join(lines)

    def get_full_instructions(self, name: str) -> str | None:
        """Tier 2: full SKILL.md instructions."""
        skill = self.get(name)
        return skill.instructions if skill else None

    def reload(self):
        """Re-scan the skills directory (for hot reloading)."""
        self.registry.clear()
        self._discover()