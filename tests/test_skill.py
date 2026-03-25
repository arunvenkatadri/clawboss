"""Tests for clawboss.skill — Skill, ToolDefinition, ToolParameter, SkillStore."""

import json

import pytest

from clawboss.skill import Skill, SkillStore, ToolDefinition, ToolParameter, _slugify


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_skill(**overrides) -> Skill:
    defaults = dict(
        name="web_research",
        description="Research topics on the web",
        triggers=["search", "research"],
        version="1.0",
        role="You are a researcher",
        task="Search the web for information",
        instructions=["Be thorough", "Cite sources"],
        examples=[{"user": "Search for Python", "assistant": "Here are results..."}],
        output_format="Markdown summary",
        tools=[
            ToolDefinition(
                name="web_search",
                description="Search the web",
                parameters=[
                    ToolParameter(name="query", type="string", description="Search query", required=True),
                    ToolParameter(name="limit", type="integer", description="Max results", default=10),
                ],
            ),
        ],
        supervision={"max_iterations": 3, "tool_timeout": 15},
    )
    defaults.update(overrides)
    return Skill(**defaults)


# ---------------------------------------------------------------------------
# ToolParameter
# ---------------------------------------------------------------------------


class TestToolParameter:
    def test_all_fields(self):
        p = ToolParameter(
            name="query",
            type="string",
            description="Search query",
            required=True,
            default=None,
        )
        assert p.name == "query"
        assert p.type == "string"
        assert p.description == "Search query"
        assert p.required is True
        assert p.default is None

    def test_defaults(self):
        p = ToolParameter(name="x")
        assert p.type == "string"
        assert p.description == ""
        assert p.required is False
        assert p.default is None


# ---------------------------------------------------------------------------
# ToolDefinition
# ---------------------------------------------------------------------------


class TestToolDefinition:
    def test_with_parameters(self):
        td = ToolDefinition(
            name="search",
            description="Search the web",
            parameters=[
                ToolParameter(name="q", type="string", required=True),
            ],
        )
        assert td.name == "search"
        assert len(td.parameters) == 1
        assert td.parameters[0].name == "q"

    def test_empty_parameters(self):
        td = ToolDefinition(name="noop", description="Does nothing")
        assert td.parameters == []


# ---------------------------------------------------------------------------
# Skill — to_dict / from_dict roundtrip
# ---------------------------------------------------------------------------


class TestSkillRoundtrip:
    def test_roundtrip(self):
        original = _make_skill()
        d = original.to_dict()
        restored = Skill.from_dict(d)
        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.triggers == original.triggers
        assert restored.version == original.version
        assert restored.role == original.role
        assert restored.task == original.task
        assert restored.instructions == original.instructions
        assert restored.examples == original.examples
        assert restored.output_format == original.output_format
        assert len(restored.tools) == len(original.tools)
        assert restored.tools[0].name == original.tools[0].name
        assert len(restored.tools[0].parameters) == len(original.tools[0].parameters)
        assert restored.supervision == original.supervision

    def test_from_dict_minimal(self):
        s = Skill.from_dict({"name": "minimal", "description": "Just a name"})
        assert s.name == "minimal"
        assert s.tools == []
        assert s.instructions == []

    def test_to_dict_omits_empty_fields(self):
        s = Skill(name="bare", description="bare skill")
        d = s.to_dict()
        assert "triggers" not in d  # empty list is falsy
        assert "instructions" not in d
        assert "tools" not in d


# ---------------------------------------------------------------------------
# Skill — to_json
# ---------------------------------------------------------------------------


class TestSkillToJson:
    def test_produces_valid_json(self):
        skill = _make_skill()
        j = skill.to_json()
        parsed = json.loads(j)
        assert parsed["name"] == "web_research"

    def test_json_contains_tools(self):
        skill = _make_skill()
        parsed = json.loads(skill.to_json())
        assert len(parsed["tools"]) == 1
        assert parsed["tools"][0]["name"] == "web_search"


# ---------------------------------------------------------------------------
# Skill — to_poml
# ---------------------------------------------------------------------------


class TestSkillToPoml:
    def test_produces_poml_with_metadata(self):
        skill = _make_skill()
        poml = skill.to_poml()
        assert "<!--" in poml
        assert "-->" in poml
        assert "metadata:" in poml
        assert "description: Research topics on the web" in poml

    def test_poml_has_triggers(self):
        skill = _make_skill()
        poml = skill.to_poml()
        assert "triggers:" in poml
        assert "search" in poml

    def test_poml_has_role_section(self):
        skill = _make_skill()
        poml = skill.to_poml()
        assert "<role>" in poml
        assert "You are a researcher" in poml
        assert "</role>" in poml

    def test_poml_has_task_section(self):
        skill = _make_skill()
        poml = skill.to_poml()
        assert "<task>" in poml
        assert "</task>" in poml

    def test_poml_has_instruction_section(self):
        skill = _make_skill()
        poml = skill.to_poml()
        assert "<instruction>" in poml
        assert "- Be thorough" in poml
        assert "- Cite sources" in poml
        assert "</instruction>" in poml

    def test_poml_has_example_section(self):
        skill = _make_skill()
        poml = skill.to_poml()
        assert "<example>" in poml
        assert "User: Search for Python" in poml
        assert "</example>" in poml

    def test_poml_has_output_format(self):
        skill = _make_skill()
        poml = skill.to_poml()
        assert "<output_format>" in poml
        assert "Markdown summary" in poml
        assert "</output_format>" in poml

    def test_poml_has_tools(self):
        skill = _make_skill()
        poml = skill.to_poml()
        assert "tools:" in poml
        assert "- name: web_search" in poml

    def test_poml_has_supervision(self):
        skill = _make_skill()
        poml = skill.to_poml()
        assert "supervision:" in poml
        assert "max_iterations: 3" in poml
        assert "tool_timeout: 15" in poml

    def test_poml_without_optional_sections(self):
        skill = Skill(name="bare", description="bare skill")
        poml = skill.to_poml()
        assert "<role>" not in poml
        assert "<task>" not in poml
        assert "<instruction>" not in poml
        assert "<example>" not in poml
        assert "<output_format>" not in poml


# ---------------------------------------------------------------------------
# _slugify
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_simple_name(self):
        assert _slugify("web_search") == "web_search"

    def test_spaces_to_underscores(self):
        assert _slugify("Web Search Tool") == "web_search_tool"

    def test_special_characters_removed(self):
        assert _slugify("search@v2.1!") == "search_v2_1"

    def test_leading_trailing_underscores_stripped(self):
        assert _slugify("  --hello--  ") == "hello"

    def test_mixed_case(self):
        assert _slugify("MyAwesomeTool") == "myawesometool"

    def test_empty_after_strip(self):
        assert _slugify("!!!") == ""


# ---------------------------------------------------------------------------
# SkillStore — CRUD
# ---------------------------------------------------------------------------


class TestSkillStoreCRUD:
    def test_save_and_get(self, tmp_path):
        store = SkillStore(str(tmp_path / "skills"))
        skill = _make_skill()
        store.save(skill)
        loaded = store.get("web_research")
        assert loaded is not None
        assert loaded.name == "web_research"
        assert loaded.description == "Research topics on the web"

    def test_get_nonexistent_returns_none(self, tmp_path):
        store = SkillStore(str(tmp_path / "skills"))
        assert store.get("nonexistent") is None

    def test_list_skills(self, tmp_path):
        store = SkillStore(str(tmp_path / "skills"))
        store.save(_make_skill(name="skill_a", description="A"))
        store.save(_make_skill(name="skill_b", description="B"))
        items = store.list()
        assert len(items) == 2
        names = {item["name"] for item in items}
        assert names == {"skill_a", "skill_b"}

    def test_list_empty_store(self, tmp_path):
        store = SkillStore(str(tmp_path / "skills"))
        assert store.list() == []

    def test_delete_existing(self, tmp_path):
        store = SkillStore(str(tmp_path / "skills"))
        store.save(_make_skill())
        assert store.delete("web_research") is True
        assert store.get("web_research") is None

    def test_delete_nonexistent(self, tmp_path):
        store = SkillStore(str(tmp_path / "skills"))
        assert store.delete("nonexistent") is False

    def test_exists(self, tmp_path):
        store = SkillStore(str(tmp_path / "skills"))
        store.save(_make_skill())
        assert store.exists("web_research") is True
        assert store.exists("nonexistent") is False

    def test_save_overwrites(self, tmp_path):
        store = SkillStore(str(tmp_path / "skills"))
        store.save(_make_skill(description="v1"))
        store.save(_make_skill(description="v2"))
        loaded = store.get("web_research")
        assert loaded.description == "v2"

    def test_creates_directory_if_missing(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "skills"
        store = SkillStore(str(deep_path))
        assert deep_path.exists()


# ---------------------------------------------------------------------------
# SkillStore — POML export
# ---------------------------------------------------------------------------


class TestSkillStoreExport:
    def test_export_poml(self, tmp_path):
        store = SkillStore(str(tmp_path / "skills"))
        store.save(_make_skill())
        poml = store.export_poml("web_research")
        assert poml is not None
        assert "<!--" in poml
        assert "web_search" in poml

    def test_export_poml_nonexistent(self, tmp_path):
        store = SkillStore(str(tmp_path / "skills"))
        assert store.export_poml("nonexistent") is None

    def test_export_all_poml(self, tmp_path):
        store = SkillStore(str(tmp_path / "skills"))
        store.save(_make_skill(name="skill_a"))
        store.save(_make_skill(name="skill_b"))
        out_dir = str(tmp_path / "poml_output")
        paths = store.export_all_poml(out_dir)
        assert len(paths) == 2
        for p in paths:
            assert p.exists()
            assert p.suffix == ".poml"
            content = p.read_text()
            assert "<!--" in content

    def test_export_all_poml_empty_store(self, tmp_path):
        store = SkillStore(str(tmp_path / "skills"))
        paths = store.export_all_poml(str(tmp_path / "poml_output"))
        assert paths == []
