from pathlib import Path


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

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.indexes_dir.mkdir(exist_ok=True)
        self.runs_dir.mkdir(exist_ok=True)
        gitignore = self.root / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("*\n")

    def index_db(self, key: str) -> Path:
        return self.indexes_dir / f"{key}.db"

    def index_meta(self, key: str) -> Path:
        return self.indexes_dir / f"{key}.json"

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id
