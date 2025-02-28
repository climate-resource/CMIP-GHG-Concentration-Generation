"""
Dependency tracking and reference capturing
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from attrs import asdict, define


@define
class SourceInfo:
    """Information about a source"""

    short_name: str
    """Short name of the source"""

    licence: str
    """Licence applied to the source's data"""

    reference: str
    """Reference for the source"""

    resource_type: str
    """Resource type of the source (used for cross-linking on Zenodo)"""

    url: str
    """URL"""

    doi: str | None = None
    """DOI"""


def ensure_source_table_exists(db_cursor: sqlite3.Connection) -> None:
    """
    Ensure that the source table exists in the database

    Parameters
    ----------
    db_cursor
        Database cursor to use for executing SQL commands
    """
    source_table_check = db_cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table'
        AND name='source';
    """).fetchall()

    if not source_table_check:
        # Create the table
        db_cursor.execute("""
            CREATE TABLE source(
                short_name VARCHAR(255) NOT NULL PRIMARY KEY,
                licence VARCHAR(255) NOT NULL,
                reference VARCHAR(4095) NOT NULL,
                resource_type VARCHAR(255) NOT NULL,
                url VARCHAR(255) NOT NULL,
                doi VARCHAR(255) NULL
            );
        """)


def save_source_info_to_db(
    db: Path,
    source_info: SourceInfo | tuple[SourceInfo, ...],
) -> None:
    """
    Save source information to the database

    Parameters
    ----------
    db
        Path to the database (connection is managed by this function for convenience)

    source_info
        Source information
    """
    db_connection = sqlite3.connect(db)
    db_connection.row_factory = sqlite3.Row

    if isinstance(source_info, SourceInfo):
        source_info_for_db = (asdict(source_info),)
    else:
        source_info_for_db = (asdict(v) for v in source_info)

    with db_connection as db_cursor:
        ensure_source_table_exists(db_cursor)

        for si in source_info_for_db:
            existing = db_cursor.execute(
                "SELECT * FROM source WHERE short_name = ?", (si["short_name"],)
            ).fetchall()

            if not existing:
                # Not in DB, insert
                db_cursor.execute(
                    """
                        INSERT
                        INTO source
                        VALUES(:short_name, :licence, :reference, :resource_type, :url, :doi)
                    """,
                    si,
                )

            elif dict(existing[0]) == si:
                # All matches, do nothing
                pass

            elif len(existing) > 1:
                # Should be impossible to get here because short_name is unique
                raise NotImplementedError

            else:
                msg = (
                    "Entry is already in the database, but with a different value. "
                    f"{dict(existing[0])=}. {si=}"
                )
                raise ValueError(msg)

    db_connection.close()


def ensure_dependency_table_exists(db_cursor: sqlite3.Connection) -> None:
    """
    Ensure that the dependency table exists in the database

    Parameters
    ----------
    db_cursor
        Database cursor to use for executing SQL commands
    """
    dep_table_check = db_cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table'
        AND name='dependencies';
    """).fetchall()

    if not dep_table_check:
        # Create the table
        db_cursor.execute("""
            CREATE TABLE dependencies(
                gas VARCHAR(255) NOT NULL,
                short_name VARCHAR(255),
                UNIQUE (gas, short_name),
                FOREIGN KEY (short_name) REFERENCES source(short_name)
            );
        """)


def save_dependency_into_db(
    db: Path,
    gas: str,
    dependency_short_name: str,
) -> None:
    """
    Save a gas' dependency into the database

    Parameters
    ----------
    db
        Path to the database (connection is managed by this function for convenience)

    gas
        Gas for which to save the dependency

    dependency_short_name
        Short name of the source on which the gas depends
    """
    db_connection = sqlite3.connect(db)
    db_connection.row_factory = sqlite3.Row

    with db_connection as db_cursor:
        ensure_source_table_exists(db_cursor)
        ensure_dependency_table_exists(db_cursor)

        existing = db_cursor.execute(
            "SELECT * FROM dependencies WHERE gas = ? AND short_name = ?", (gas, dependency_short_name)
        ).fetchall()

        if not existing:
            # Not in DB, insert
            db_cursor.execute("INSERT INTO dependencies VALUES(?, ?)", (gas, dependency_short_name))

        elif dict(existing[0]) == {"gas": gas, "short_name": dependency_short_name}:
            # All matches, do nothing
            pass

        elif len(existing) > 1:
            # Should be impossible to get here because gas - short name combination is unique
            raise NotImplementedError

        else:
            new = {"gas": gas, "short_name": dependency_short_name}
            msg = (
                "Entry is already in the database, but with a different value. "
                f"{dict(existing[0])=}. {new=}"
            )
            raise ValueError(msg)

    db_connection.close()
