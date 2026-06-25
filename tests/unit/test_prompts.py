from rag_lab.prompts import DEFAULT_SYSTEM_INSTRUCTIONS, PromptBuilder


def test_build_uses_default_instructions_when_unset():
    prompt = PromptBuilder().build(question="q", results=[])
    assert DEFAULT_SYSTEM_INSTRUCTIONS.strip() in prompt


def test_build_uses_custom_instructions():
    prompt = PromptBuilder(system_instructions="ONLY speak in haiku.").build(
        question="q", results=[]
    )
    assert "ONLY speak in haiku." in prompt
    assert DEFAULT_SYSTEM_INSTRUCTIONS.strip() not in prompt
