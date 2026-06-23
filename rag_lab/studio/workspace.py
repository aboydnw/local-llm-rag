from pathlib import Path


def _safe_segment(name: str) -> str:
    if (
        not name
        or name in (".", "..")
        or "/" in name
        or "\\" in name
        or Path(name).is_absolute()
    ):
        raise ValueError(f"unsafe path segment: {name!r}")
    return name


class Workspace:
    """Owns the studio's local `.rag-lab/` working directory."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @classmethod
    def default(cls) -> "Workspace":
        return cls(Path(".rag-lab"))

    @property
    def indexes_dir(self) -> Path:
        return self.root / "indexes"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def corpora_dir(self) -> Path:
        return self.root / "corpora"

    @property
    def clones_dir(self) -> Path:
        return self.root / "clones"

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.indexes_dir.mkdir(exist_ok=True)
        self.runs_dir.mkdir(exist_ok=True)
        self.corpora_dir.mkdir(exist_ok=True)
        self.clones_dir.mkdir(exist_ok=True)
        gitignore = self.root / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("*\n")

    def index_db(self, key: str) -> Path:
        return self.indexes_dir / f"{_safe_segment(key)}.db"

    def index_meta(self, key: str) -> Path:
        return self.indexes_dir / f"{_safe_segment(key)}.json"

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / _safe_segment(run_id)

    def corpus_file(self, name: str) -> Path:
        return self.corpora_dir / f"{_safe_segment(name)}.json"

    def clone_dir(self, key: str) -> Path:
        return self.clones_dir / _safe_segment(key)
