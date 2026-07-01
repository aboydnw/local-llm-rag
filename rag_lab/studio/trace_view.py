import streamlit as st


def render_steps(steps: list[dict], *, key_prefix: str) -> None:
    """Render agent trace steps: one expander per tool call, prompt behind a checkbox."""
    for i, step in enumerate(steps, start=1):
        if step.get("action") is None:
            st.markdown(f"**{i}. Ready to answer** — {step.get('thought', '')}")
            continue
        with st.expander(f"{i}. {step['action']}: {step['action_input']}"):
            if step.get("thought"):
                st.markdown(f"**Thought:** {step['thought']}")
            st.text((step.get("observation") or "")[:2000])
            if step.get("prompt") and st.checkbox(
                "Show prompt sent this step", key=f"{key_prefix}_stepprompt_{i}"
            ):
                st.code(step["prompt"], language="text")
