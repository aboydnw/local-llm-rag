from pathlib import Path

import streamlit as st

from rag_lab.eval.golden_set import GoldenItem
from rag_lab.studio import golden_io


def render() -> None:
    st.title("Golden set")
    path = Path(st.session_state["golden"])
    items = golden_io.load_items(path)

    st.caption("Editing the golden set makes new runs non-comparable to older ones.")
    st.dataframe(
        [{"id": i.id, "question": i.question,
          "ideal_docs": ", ".join(i.ideal_docs),
          "must_mention": ", ".join(i.must_mention)} for i in items],
        use_container_width=True,
    )

    st.subheader("Add / edit an item")
    ids = ["(new)"] + [i.id for i in items]
    pick = st.selectbox("Item", ids)
    existing = next((i for i in items if i.id == pick), None)

    item_id = st.text_input("id", value=existing.id if existing else "")
    question = st.text_area("question", value=existing.question if existing else "")
    ideal_docs = st.text_area("ideal_docs (one per line)",
                              value="\n".join(existing.ideal_docs) if existing else "")
    must_mention = st.text_area("must_mention (one per line)",
                                value="\n".join(existing.must_mention) if existing else "")
    ideal_answer = st.text_area("ideal_answer", value=existing.ideal_answer if existing else "")

    c1, c2 = st.columns(2)
    if c1.button("Save item", type="primary") and item_id and question:
        item = GoldenItem(
            id=item_id,
            question=question,
            ideal_docs=[s for s in ideal_docs.splitlines() if s.strip()],
            must_mention=[s for s in must_mention.splitlines() if s.strip()],
            ideal_answer=ideal_answer,
        )
        golden_io.save_items(path, golden_io.upsert_item(items, item))
        st.success(f"Saved {item_id} to {path}")
        st.rerun()
    if c2.button("Delete item") and existing:
        golden_io.save_items(path, golden_io.delete_item(items, existing.id))
        st.success(f"Deleted {existing.id}")
        st.rerun()
