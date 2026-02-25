"""
Skill Executor — runs skills safely in a sandboxed subprocess.

Two types:
1. Script-based: runs run.py or run.sh in a restricted environment
2. Instruction-only: returns SKILL.md for Claude to follow
"""

import subprocess
import json
from pathlib import Path

from agent.skills.loader import SkillRegistry


def execute_skill_action(
    registry: SkillRegistry,
    skill_name: str,
    parameters: dict,
    timeout: int = 30
) -> str:
    """Execute a skill by name."""
    skill = registry.get(skill_name)
    if not skill:
        available = ", ".join(registry.registry.keys()) or "none"
        return f"Skill '{skill_name}' not found. Available: {available}"

    # Check for runnable scripts
    script_path = skill.path / "run.py"
    if script_path.exists():
        return _run_python_script(script_path, parameters, skill.path, timeout)

    shell_script = skill.path / "run.sh"
    if shell_script.exists():
        return _run_shell_script(shell_script, parameters, skill.path, timeout)

    # No script — return instructions for Claude to follow
    return (
        f"## Skill: {skill.name}\n\n"
        f"{skill.instructions}\n\n"
        f"Parameters: {json.dumps(parameters, indent=2)}\n\n"
        f"Follow the instructions above to complete this task."
    )

def _run_python_script(
    script_path: Path,
    parameters: dict,
    cwd: Path,
    timeout: int
) -> str:
    """Run a Python script in a sandboxed subprocess."""
    try:
        result = subprocess.run(
            ["python", str(script_path)],
            input=json.dumps(parameters),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd),
            env={
                "PATH": "/usr/bin:/usr/local/bin",
                "HOME": str(cwd),
            },
        )

        output = result.stdout.strip()
        errors = result.stderr.strip()

        if result.returncode != 0:
            return f"Script failed (exit {result.returncode}):\n{errors}"

        return output if output else "Script completed."

    except subprocess.TimeoutExpired:
        return f"Script timed out after {timeout}s."
    except Exception as e:
        return f"Script error: {e}"


def _run_shell_script(
    script_path: Path,
    parameters: dict,
    cwd: Path,
    timeout: int
) -> str:
    """Run a shell script in a sandboxed subprocess."""
    try:
        env = {"PATH": "/usr/bin:/usr/local/bin", "HOME": str(cwd)}

        for key, value in parameters.items():
            safe_key = key.upper().replace(" ", "_")
            env[f"PARAM_{safe_key}"] = str(value)

        result = subprocess.run(
            ["bash", str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd),
            env=env,
        )

        output = result.stdout.strip()
        errors = result.stderr.strip()

        if result.returncode != 0:
            return f"Script failed:\n{errors}"

        return output if output else "Script completed."

    except subprocess.TimeoutExpired:
        return f"Script timed out after {timeout}s."
    except Exception as e:
        return f"Script error: {e}"