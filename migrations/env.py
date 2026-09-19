"""Alembic environment for the Research Gateway.

Lives at the repo-root `migrations/` per docs/ARCHITECTURE.md §17, but the
gateway's `alembic.ini`/settings/models live under
`services/research-gateway/`. We add that directory to `sys.path` explicitly
(rather than relying on alembic.ini's `prepend_sys_path`, which is sensitive
to the invoking cwd) so `import config.settings` / `import domain.models` work
regardless of where `alembic` is invoked from.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

_SERVICE_DIR = Path(__file__).resolve().parents[1] / "services" / "research-gateway"
if str(_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVICE_DIR))

from domain.models import Base  # noqa: E402
from config.settings import get_settings  # noqa: E402

# Alembic Config object, providing access to values in alembic.ini.
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for autogenerate support.
target_metadata = Base.metadata


def _database_url() -> str:
    """Resolve the SQLite URL from the same Settings the FastAPI app uses.

    `research_gateway_db_path` is a filesystem path (relative paths are
    resolved against the current process's cwd, matching how the app itself
    opens the DB), turned into a `sqlite:///` URL.
    """
    settings = get_settings()
    db_path = Path(settings.research_gateway_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path}"


def run_migrations_offline() -> None:
    url = _database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = _database_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
