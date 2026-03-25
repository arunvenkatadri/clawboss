"""Skill definitions — framework-agnostic skill format with file-based CRUD."""

import json
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ToolParameter:
    """A parameter on a tool."""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = False
    default: Any = None


@dataclass
class ToolDefinition:
    """A tool that a skill can use."""

    name: str
    description: str = ""
    parameters: List[ToolParameter] = field(default_factory=list)


@dataclass
class Skill:
    """A complete skill definition — what the agent does, how, and within what limits.

    This is the clawboss-native skill format. It can be converted to/from
    POML, JSON, YAML, or any other format your agent framework uses.
    """

    name: str
    description: str
    triggers: List[str] = field(default_factory=list)
    version: str = "1.0"

    # What the agent should do
    role: str = ""
    task: str = ""
    instructions: List[str] = field(default_factory=list)
    examples: List[Dict[str, str]] = field(default_factory=list)
    output_format: str = ""

    # What tools are available
    tools: List[ToolDefinition] = field(default_factory=list)

    # Supervision limits (maps directly to Policy.from_dict)
    supervision: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        d = asdict(self)
        # Drop empty/default fields for cleaner output
        return {k: v for k, v in d.items() if v}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Skill":
        """Create a Skill from a dictionary."""
        tools = []
        for t in d.get("tools", []):
            params = [ToolParameter(**p) for p in t.get("parameters", [])]
            tools.append(
                ToolDefinition(
                    name=t["name"],
                    description=t.get("description", ""),
                    parameters=params,
                )
            )

        return cls(
            name=d["name"],
            description=d.get("description", ""),
            triggers=d.get("triggers", []),
            version=d.get("version", "1.0"),
            role=d.get("role", ""),
            task=d.get("task", ""),
            instructions=d.get("instructions", []),
            examples=d.get("examples", []),
            output_format=d.get("output_format", ""),
            tools=tools,
            supervision=d.get("supervision", {}),
        )

    def to_poml(self) -> str:
        """Render as POML format (compatible with moneypenny and other POML loaders)."""
        lines = ["<!--"]
        lines.append("metadata:")
        lines.append(f"  description: {self.description}")
        if self.triggers:
            lines.append(f"  triggers: {', '.join(self.triggers)}")
        if self.version:
            lines.append(f"  version: {self.version}")

        if self.tools:
            lines.append("")
            lines.append("tools:")
            for tool in self.tools:
                lines.append(f"  - name: {tool.name}")
                if tool.description:
                    lines.append(f"    description: {tool.description}")
                if tool.parameters:
                    lines.append("    parameters:")
                    for p in tool.parameters:
                        lines.append(f"      {p.name}:")
                        lines.append(f"        type: {p.type}")
                        if p.description:
                            lines.append(f"        description: {p.description}")
                        if p.required:
                            lines.append("        required: true")
                        if p.default is not None:
                            lines.append(f"        default: {p.default}")

        if self.supervision:
            lines.append("")
            lines.append("supervision:")
            for k, v in self.supervision.items():
                if isinstance(v, list):
                    lines.append(f"  {k}:")
                    for item in v:
                        lines.append(f"    - {item}")
                else:
                    lines.append(f"  {k}: {v}")

        lines.append("-->")
        lines.append("")

        if self.role:
            lines.append(f"<role>\n{self.role}\n</role>")
            lines.append("")

        if self.task:
            lines.append(f"<task>\n{self.task}\n</task>")
            lines.append("")

        if self.instructions:
            lines.append("<instruction>")
            for inst in self.instructions:
                lines.append(f"- {inst}")
            lines.append("</instruction>")
            lines.append("")

        for ex in self.examples:
            lines.append("<example>")
            if "user" in ex:
                lines.append(f"User: {ex['user']}")
            if "assistant" in ex:
                lines.append(f"Assistant: {ex['assistant']}")
            lines.append("</example>")
            lines.append("")

        if self.output_format:
            lines.append(f"<output_format>\n{self.output_format}\n</output_format>")
            lines.append("")

        return "\n".join(lines)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


def _slugify(name: str) -> str:
    """Convert a skill name to a safe filename slug."""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


class SkillStore:
    """File-based CRUD for skills. Each skill is a JSON file on disk.

    Usage:
        store = SkillStore("~/.clawboss/skills")
        store.save(skill)
        skill = store.get("web_research")
        all_skills = store.list()
        store.delete("web_research")
    """

    def __init__(self, directory: str):
        self._dir = Path(os.path.expanduser(directory))
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def directory(self) -> Path:
        return self._dir

    def _path(self, name: str) -> Path:
        return self._dir / f"{_slugify(name)}.json"

    def save(self, skill: Skill) -> Path:
        """Save a skill to disk. Overwrites if exists. Returns file path."""
        path = self._path(skill.name)
        with self._lock:
            path.write_text(skill.to_json())
        return path

    def get(self, name: str) -> Optional[Skill]:
        """Load a skill by name. Returns None if not found."""
        path = self._path(name)
        if not path.exists():
            return None
        with self._lock:
            data = json.loads(path.read_text())
        return Skill.from_dict(data)

    def list(self) -> List[Dict[str, str]]:
        """List all skills (name + description). Lightweight — doesn't load full skills."""
        skills = []
        with self._lock:
            for path in sorted(self._dir.glob("*.json")):
                try:
                    data = json.loads(path.read_text())
                    skills.append(
                        {
                            "name": data.get("name", path.stem),
                            "description": data.get("description", ""),
                            "version": data.get("version", ""),
                        }
                    )
                except (json.JSONDecodeError, KeyError):
                    continue
        return skills

    def delete(self, name: str) -> bool:
        """Delete a skill. Returns True if deleted, False if not found."""
        path = self._path(name)
        with self._lock:
            if path.exists():
                path.unlink()
                return True
        return False

    def exists(self, name: str) -> bool:
        """Check if a skill exists."""
        return self._path(name).exists()

    def export_poml(self, name: str) -> Optional[str]:
        """Export a skill as POML format. Returns None if not found."""
        skill = self.get(name)
        if skill is None:
            return None
        return skill.to_poml()

    def export_all_poml(self, output_dir: str) -> List[Path]:
        """Export all skills as .poml files to a directory."""
        out = Path(os.path.expanduser(output_dir))
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for info in self.list():
            skill = self.get(info["name"])
            if skill:
                path = out / f"{_slugify(skill.name)}.poml"
                path.write_text(skill.to_poml())
                paths.append(path)
        return paths
