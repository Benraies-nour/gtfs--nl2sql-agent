"""Prompt loading: every placeholder must be filled."""
import pytest

from soretrak.prompts import knowledge, render


def test_render_fills_placeholders():
    text = render("router.md", perimetre="PERIMETRE-TEST")
    assert "PERIMETRE-TEST" in text and "{{" not in text


def test_render_refuses_missing_placeholder():
    with pytest.raises(ValueError, match="perimetre"):
        render("router.md")


def test_prompt_without_placeholder():
    assert "Jamais d'emojis" in render("synthesizer.md")


def test_knowledge_files_exist():
    for name in ("perimetre.md", "schema.md", "examples.yaml"):
        assert knowledge(name).strip()
