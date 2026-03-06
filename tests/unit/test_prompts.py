"""
Unit tests for generation.prompts (versioned registry)
"""

import pytest

from generation.prompts import (
    PROMPT_REGISTRY,
    _REQUIRED_KEYS,
    _SYSTEM_INTENT_MAP,
    active_version,
    get_prompt,
    get_system_prompt,
    list_versions,
    validate_version,
)


# ── Registry integrity ────────────────────────────────────────────────────────

class TestRegistryIntegrity:
    def test_v1_registered(self):
        assert "v1" in PROMPT_REGISTRY

    def test_v2_registered(self):
        assert "v2" in PROMPT_REGISTRY

    @pytest.mark.parametrize("version", ["v1", "v2"])
    def test_all_required_keys_present(self, version):
        missing = _REQUIRED_KEYS - PROMPT_REGISTRY[version].keys()
        assert not missing, (
            f"Version '{version}' is missing keys: {sorted(missing)}"
        )

    @pytest.mark.parametrize("version", ["v1", "v2"])
    def test_all_prompts_are_non_empty_strings(self, version):
        for key, value in PROMPT_REGISTRY[version].items():
            assert isinstance(value, str), (
                f"Version '{version}' key '{key}' is not a string"
            )
            assert value.strip(), (
                f"Version '{version}' key '{key}' is empty"
            )

    def test_system_intent_map_covers_all_intents(self):
        expected = {
            "regulatory_query", "financial_data",
            "general_finreg", "out_of_scope",
        }
        assert set(_SYSTEM_INTENT_MAP.keys()) == expected


# ── validate_version ─────────────────────────────────────────────────────────

class TestValidateVersion:
    def test_valid_version_passes(self):
        validate_version("v1")  # should not raise

    def test_unknown_version_raises(self):
        with pytest.raises(ValueError, match="not found"):
            validate_version("v99")

    def test_error_lists_available_versions(self):
        with pytest.raises(ValueError) as exc_info:
            validate_version("v99")
        assert "v1" in str(exc_info.value)


# ── list_versions ─────────────────────────────────────────────────────────────

class TestListVersions:
    def test_returns_sorted_list(self):
        versions = list_versions()
        assert versions == sorted(versions)

    def test_includes_v1_and_v2(self):
        versions = list_versions()
        assert "v1" in versions
        assert "v2" in versions


# ── get_prompt ────────────────────────────────────────────────────────────────

class TestGetPrompt:
    def test_fetches_correct_prompt(self):
        prompt = get_prompt("sys_regulatory", version="v1")
        assert isinstance(prompt, str)
        assert len(prompt) > 50  # sanity: not trivially short

    def test_different_versions_differ(self):
        p1 = get_prompt("sys_regulatory", version="v1")
        p2 = get_prompt("sys_regulatory", version="v2")
        assert p1 != p2

    def test_unknown_version_raises_key_error(self):
        with pytest.raises(KeyError):
            get_prompt("sys_regulatory", version="v99")

    def test_unknown_key_raises_key_error(self):
        with pytest.raises(KeyError):
            get_prompt("nonexistent_key", version="v1")

    def test_defaults_to_active_version(self, monkeypatch):
        from config import settings as s_module
        monkeypatch.setattr(s_module.settings, "prompt_version", "v1")
        prompt = get_prompt("sys_regulatory")
        assert prompt == get_prompt("sys_regulatory", version="v1")


# ── get_system_prompt ─────────────────────────────────────────────────────────

class TestGetSystemPrompt:
    @pytest.mark.parametrize("intent", [
        "regulatory_query", "financial_data",
        "general_finreg", "out_of_scope",
    ])
    def test_all_intents_return_prompt(self, intent):
        prompt = get_system_prompt(intent, version="v1")
        assert isinstance(prompt, str) and prompt.strip()

    def test_unknown_intent_falls_back_to_general(self):
        prompt = get_system_prompt("completely_unknown", version="v1")
        general = get_system_prompt("general_finreg", version="v1")
        assert prompt == general


# ── JSON schema embedded in prompts ──────────────────────────────────────────

class TestPromptJsonSchema:
    @pytest.mark.parametrize("version", ["v1", "v2"])
    def test_system_prompts_contain_json_instruction(self, version):
        for intent in ["regulatory_query", "financial_data",
                       "general_finreg", "out_of_scope"]:
            prompt = get_system_prompt(intent, version=version)
            assert "json" in prompt.lower(), (
                f"{version}/{intent} prompt missing JSON instruction"
            )

    @pytest.mark.parametrize("version", ["v1", "v2"])
    def test_system_prompts_contain_confidence_field(self, version):
        for intent in ["regulatory_query", "financial_data"]:
            prompt = get_system_prompt(intent, version=version)
            assert "confidence" in prompt

    @pytest.mark.parametrize("version", ["v1", "v2"])
    def test_system_prompts_contain_follow_up_field(self, version):
        for intent in ["regulatory_query", "financial_data"]:
            prompt = get_system_prompt(intent, version=version)
            assert "follow_up" in prompt


# ── User prompt templates ─────────────────────────────────────────────────────

class TestUserPromptTemplates:
    @pytest.mark.parametrize("version", ["v1", "v2"])
    def test_user_with_context_format_string(self, version):
        tpl = get_prompt("user_with_context", version=version)
        rendered = tpl.format(context_blocks="[Source 1] text", query="Q?")
        assert "[Source 1] text" in rendered
        assert "Q?" in rendered

    @pytest.mark.parametrize("version", ["v1", "v2"])
    def test_user_with_web_format_string(self, version):
        tpl = get_prompt("user_with_web", version=version)
        rendered = tpl.format(
            web_query="search", web_results="result", query="Q?"
        )
        assert "result" in rendered
        assert "Q?" in rendered

    @pytest.mark.parametrize("version", ["v1", "v2"])
    def test_user_no_context_format_string(self, version):
        tpl = get_prompt("user_no_context", version=version)
        rendered = tpl.format(query="What is MiFID II?")
        assert "What is MiFID II?" in rendered
