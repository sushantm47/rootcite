import pytest

from rootcite.retrieval.chunking import chunk_markdown

DOC = """# Disk full

Intro paragraph.

## Symptoms

Writes fail.

## Mitigation

1. Delete old logs.

```bash
# this is a comment, not a heading
du -sh /var/log
```
"""


def test_chunks_carry_heading_path_as_context():
    chunks = chunk_markdown(DOC, source="runbooks/disk.md", doc_type="runbook")
    titles = [c.title for c in chunks]
    assert titles == ["Disk full", "Disk full > Symptoms", "Disk full > Mitigation"]
    assert chunks[1].text.startswith("Disk full > Symptoms\n")


def test_hash_inside_code_fence_is_not_a_heading():
    chunks = chunk_markdown(DOC, source="runbooks/disk.md", doc_type="runbook")
    assert "# this is a comment" in chunks[-1].text


def test_ids_are_deterministic_for_idempotent_upserts():
    a = chunk_markdown(DOC, source="runbooks/disk.md", doc_type="runbook")
    b = chunk_markdown(DOC, source="runbooks/disk.md", doc_type="runbook")
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)


def test_long_sections_are_split_under_limit():
    body = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(20))
    chunks = chunk_markdown(f"# Big\n\n{body}", source="big.md", doc_type="doc", max_chars=500)
    assert len(chunks) > 1
    heading_len = len("Big\n")
    assert all(len(c.text) <= 500 + heading_len for c in chunks)


def test_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_markdown(DOC, source="x.md", doc_type="doc", max_chars=100, overlap=100)
