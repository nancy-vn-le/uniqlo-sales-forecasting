"""SQLAlchemy engine helper for uniqlo.db."""

from pathlib import Path
from sqlalchemy import create_engine, Engine


def get_engine() -> Engine:
    root = Path(__file__).parents[2]
    db_path = root / "data" / "uniqlo.db"
    if not db_path.exists():
        db_path = root / "data" / "demo.db"
    return create_engine(f"sqlite:///{db_path}", echo=False)
