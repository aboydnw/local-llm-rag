from pathlib import Path

import streamlit as st

from rag_lab.eval.golden_set import GoldenItem
from rag_lab.studio import golden_io


def render() -> None:
    """Render the evaluation-set viewer and question editor."""
    st.title("Test questions")
    st.caption("Define the questions and evidence used to measure every evaluation run.")
    with st.expander("Evaluation set file", expanded=False):
        st.session_state["golden"] = st.text_input(
            "YAML path",
            value=st.session_state["golden"],
            help="The local YAML file where these test questions are stored.",
        )
    path = Path(st.session_state["golden"])
    items = golden_io.load_items(path)
    st.warning("Changing these questions makes new results non-comparable with older reports.")
    if items:
        st.dataframe(
            [
                {
                    "ID": item.id,
                    "Question": item.question,
                    "Expected documents": ", ".join(item.ideal_docs),
                    "Required phrases": ", ".join(item.must_mention),
                }
                for item in items
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No test questions yet. Add the first one below.")

    st.subheader("Add or edit a test question")
    ids = ["New question", *[item.id for item in items]]
    pick = st.selectbox("Question", ids)
    existing = next((item for item in items if item.id == pick), None)
    item_id = st.text_input("ID", value=existing.id if existing else "")
    question = st.text_area("Question text", value=existing.question if existing else "")
    ideal_docs = st.text_area(
        "Expected documents · one per line",
        value="\n".join(existing.ideal_docs) if existing else "",
    )
    must_mention = st.text_area(
        "Required answer phrases · one per line",
        value="\n".join(existing.must_mention) if existing else "",
    )
    ideal_answer = st.text_area(
        "Ideal answer · optional", value=existing.ideal_answer if existing else ""
    )
    save, remove = st.columns(2)
    if save.button("Save question", type="primary") and item_id.strip() and question.strip():
        normalized_id = item_id.strip()
        original_id = existing.id if existing else None
        if normalized_id != original_id and any(item.id == normalized_id for item in items):
            st.error(f"A question with ID '{normalized_id}' already exists.")
        else:
            item = GoldenItem(
                id=normalized_id,
                question=question.strip(),
                ideal_docs=[line.strip() for line in ideal_docs.splitlines() if line.strip()],
                must_mention=[line.strip() for line in must_mention.splitlines() if line.strip()],
                ideal_answer=ideal_answer.strip(),
            )
            updated = golden_io.upsert_item(items, item)
            if original_id is not None and original_id != normalized_id:
                updated = golden_io.delete_item(updated, original_id)
            golden_io.save_items(path, updated)
            st.toast(f"Saved {normalized_id}")
            st.rerun()
    confirmed = remove.checkbox(
        "Confirm delete",
        disabled=existing is None,
        key=f"confirm-delete-{existing.id}" if existing else "confirm-delete-new",
    )
    if remove.button("Delete question", disabled=existing is None or not confirmed):
        golden_io.save_items(path, golden_io.delete_item(items, existing.id))
        st.rerun()
