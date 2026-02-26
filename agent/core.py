"""
Core Agent — the brain that ties memory, skills, and tools together.

This module:
1. Builds the system prompt from your identity files
2. Defines tools that Claude can call
3. Runs the agentic loop (Claude calls tools → we execute → repeat)
4. Handles permission gating for risky actions
5. Logs conversations to daily session files
"""

import json
import datetime
from pathlib import Path

import anthropic

from agent.config import Config
from agent.memory.loader import load_identity
from agent.memory.store import MemoryStore
from agent.skills.loader import SkillRegistry

class Agent:
    """
    The core agent. Create one, then call agent.chat("your message")
    to get a response.
    """

    def __init__(self, config: Config):
        self.config = config

        # Create the Anthropic API client
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)

        # Initialize the memory system
        self.memory = MemoryStore(config.database_path, config.memory_dir)

        # Initialize the skill registry
        self.skills = SkillRegistry(config.skills_dir)

        # Load identity files (SOUL.md, USER.md, MEMORY.md)
        self.identity = load_identity(config.memory_dir)

        # Conversation history — this is what makes it multi-turn
        self.conversation_history: list[dict] = []

        # Ensure daily log directory exists
        daily_dir = config.memory_dir / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)

    def build_system_prompt(self) -> str:
        """Build the system prompt that tells Claude who it is."""
        skill_descriptions = self.skills.describe_all()

        return f"""You are a personal AI agent. You act on behalf of your user.
You have persistent memory and can learn over time.

{self.identity}

## Available Skills
{skill_descriptions}

## Available Tools
You have these tools:
- **search_memory**: Search your memory for relevant context
- **save_memory**: Save something important to remember
- **read_file**: Read a file from the filesystem
- **write_file**: Write content to a file (REQUIRES PERMISSION)
- **run_skill**: Execute a registered skill
- **list_skills**: See all available skills

## Guidelines
- Always check memory before asking the user something they may have told you before.
- When you learn something new about the user, save it to memory.
- Be proactive — suggest actions, anticipate needs.
- For risky actions (file writes, sending messages), explain what you'll do first.
- Keep responses concise and actionable.

Today is {datetime.date.today().isoformat()}.
"""
    
    def get_tools(self) -> list[dict]:
        """Define the tool schemas that Claude can call."""
        return [
            {
                "name": "search_memory",
                "description": (
                    "Search your persistent memory for relevant context. "
                    "Uses hybrid vector + keyword search. Use this BEFORE "
                    "asking the user something — they may have told you before."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results (default 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "save_memory",
                "description": (
                    "Save important information to persistent memory. "
                    "Use for decisions, preferences, lessons, and facts "
                    "you learn about the user."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "What to remember"
                        },
                        "category": {
                            "type": "string",
                            "enum": ["decision", "preference", "lesson", "fact", "context"],
                            "description": "Category of this memory"
                        }
                    },
                    "required": ["content", "category"]
                }
            },
            {
                "name": "read_file",
                "description": "Read a file from the local filesystem.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file"
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "write_file",
                "description": (
                    "Write content to a file. REQUIRES USER APPROVAL. "
                    "Always explain what you're writing and why."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to write to"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write"
                        }
                    },
                    "required": ["path", "content"]
                }
            },
            {
                "name": "run_skill",
                "description": "Execute a registered local skill by name.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "Name of the skill"
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Parameters for the skill"
                        }
                    },
                    "required": ["skill_name"]
                }
            },
            {
                "name": "list_skills",
                "description": "List all available skills.",
                "input_schema": {
                    "type": "object",
                    "properties": {}
                }
            },
        ]
    
    def handle_tool_call(
        self,
        tool_name: str,
        tool_input: dict,
        approval_callback=None
    ) -> str:
        """Execute a tool call and return the result."""
        # ── Permission Gate ──────────────────────────────
        if tool_name not in self.config.auto_approve_actions:
            if approval_callback:
                description = (
                    f"Tool: {tool_name}\n"
                    f"Input: {json.dumps(tool_input, indent=2)}"
                )
                approved = approval_callback(description)
                if not approved:
                    return "Action denied by user."
            else:
                return f"Action '{tool_name}' requires approval."

        # ── Route to the right handler ────────────────────
        if tool_name == "search_memory":
            results = self.memory.search(
                query=tool_input["query"],
                limit=tool_input.get("limit", 5)
            )
            if not results:
                return "No relevant memories found."
            return "\n\n".join(
                f"[{r['category']}] (score: {r['score']:.2f}) {r['content']}"
                for r in results
            )

        elif tool_name == "save_memory":
            self.memory.save(
                content=tool_input["content"],
                category=tool_input["category"]
            )
            return f"Saved to memory under '{tool_input['category']}'."

        elif tool_name == "read_file":
            path = Path(tool_input["path"]).resolve()
            if not path.exists():
                return f"File not found: {path}"
            try:
                return path.read_text(encoding="utf-8")[:10000]
            except Exception as e:
                return f"Error reading file: {e}"

        elif tool_name == "write_file":
            path = Path(tool_input["path"]).resolve()
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(tool_input["content"], encoding="utf-8")
                return f"Written to {path}"
            except Exception as e:
                return f"Error writing file: {e}"

        elif tool_name == "run_skill":
            from agent.skills.executor import execute_skill_action
            return execute_skill_action(
                self.skills,
                tool_input["skill_name"],
                tool_input.get("parameters", {}),
                timeout=self.config.skill_timeout_seconds
            )

        elif tool_name == "list_skills":
            return self.skills.describe_all()

        else:
            return f"Unknown tool: {tool_name}"
        
    def chat(self, user_message: str, approval_callback=None) -> str:
        """Send a message to the agent and get a response."""
        # Add user message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        system_prompt = self.build_system_prompt()
        tools = self.get_tools()

        # ── The Agentic Loop ─────────────────────────────
        while True:
            # Call the Anthropic API
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=4096,
                system=system_prompt,
                tools=tools, # type: ignore
                messages=self.conversation_history # type: ignore
            )

            # Add Claude's response to history
            assistant_content = response.content
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_content
            })

            # Check: did Claude call any tools?
            tool_use_blocks = [
                block for block in assistant_content
                if block.type == "tool_use"
            ]

            # If NO tool calls → Claude is done, return the text
            if not tool_use_blocks:
                text_blocks = [
                    block.text for block in assistant_content
                    if block.type == "text"
                ]
                final_response = "\n".join(text_blocks)

                # Log this exchange
                self._log_session(user_message, final_response)

                return final_response

            # If there ARE tool calls → execute them and loop back
            tool_results = []
            for tool_block in tool_use_blocks:
                result = self.handle_tool_call(
                    tool_block.name,
                    tool_block.input,
                    approval_callback=approval_callback
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": result
                })

            # Add tool results to history and continue the loop
            self.conversation_history.append({
                "role": "user",
                "content": tool_results
            })

    def _log_session(self, user_msg: str, agent_msg: str):
        """Write this exchange to today's daily log."""
        today = datetime.date.today().isoformat()
        log_path = self.config.memory_dir / "daily" / f"{today}.md"

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        entry = (
            f"\n## {timestamp}\n"
            f"**User:** {user_msg}\n\n"
            f"**Agent:** {agent_msg}\n\n"
            f"---\n"
        )

        with open(log_path, "a", encoding="utf-8") as f:
            if log_path.stat().st_size == 0:
                f.write(f"# Session Log — {today}\n")
            f.write(entry)

    def reset_conversation(self):
        """Clear conversation history (start fresh)."""
        self.conversation_history = []