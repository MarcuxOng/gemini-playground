from __future__ import annotations

from collections.abc import Generator
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


def _sanitize_db_url(url: str) -> str:
    """Strip unsupported params (e.g. channel_binding) and ensure sslmode=require."""
    parsed = urlparse(url.strip())
    params = parse_qs(parsed.query, keep_blank_values=True)
    params.pop("channel_binding", None)
    if "sslmode" not in params:
        params["sslmode"] = ["require"]
    clean_query = urlencode({k: v[0] for k, v in params.items()})
    return urlunparse(parsed._replace(query=clean_query))


DATABASE_URL: str = settings.database_url or "sqlite:///./app.db"

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(_sanitize_db_url(DATABASE_URL))

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


# Dependency to get a database session
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
