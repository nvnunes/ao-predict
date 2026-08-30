"""Generated option-table helpers for simulation initialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from astropy import units as u

from .api import TableOptionsConfig


@dataclass(frozen=True)
class GeneratedOptions:
    """Generated table-form options for dataset initialization.

    Attributes:
        columns: Ordered option table column names.
        rows: Table rows, one row per simulation.
        units: Units keyed by physical table column name.
        broadcast: Scalar or structured option defaults applied to every row.
    """

    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    units: Mapping[str, str | u.UnitBase] = field(default_factory=dict)
    broadcast: Mapping[str, Any] = field(default_factory=dict)

    def to_table_options_config(self) -> TableOptionsConfig:
        """Return the generated options as an initialization config object."""
        return TableOptionsConfig(
            broadcast=dict(self.broadcast),
            columns=list(self.columns),
            units=dict(self.units),
            rows=[list(row) for row in self.rows],
        )


def options_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    broadcast: Mapping[str, Any] | None = None,
    columns: Sequence[str] | None = None,
    units: Mapping[str, str | u.UnitBase] | None = None,
) -> GeneratedOptions:
    """Build generated table options from row mappings.

    Args:
        rows: Per-simulation option rows. Each row must provide the same keys,
            unless explicit ``columns`` select the required subset.
        broadcast: Optional defaults applied through ``TableOptionsConfig``.
        columns: Optional explicit column order.
        units: Units keyed by physical table column name.

    Returns:
        Generated table options ready to pass into ``InitDatasetRequest``.

    Raises:
        ValueError: If rows are empty, column names are invalid, or a row is
            missing a required column.
    """
    row_list = [dict(row) for row in rows]
    if not row_list:
        raise ValueError("options_from_rows requires at least one row.")

    if columns is None:
        column_tuple = tuple(row_list[0].keys())
        if not column_tuple:
            raise ValueError("Generated option rows must contain at least one column.")
        for index, row in enumerate(row_list[1:], start=1):
            if tuple(row.keys()) != column_tuple:
                raise ValueError(
                    "Generated option rows must have identical key order when "
                    "explicit columns are not provided "
                    f"(row {index} differs)."
                )
    else:
        column_tuple = tuple(str(column) for column in columns)
        if not column_tuple:
            raise ValueError("Explicit generated option columns must not be empty.")
        duplicate_columns = sorted(
            {column for column in column_tuple if column_tuple.count(column) > 1}
        )
        if duplicate_columns:
            raise ValueError(f"Generated option columns contain duplicates: {duplicate_columns}.")
        for column in column_tuple:
            if not column:
                raise ValueError("Generated option columns must be non-empty strings.")

    output_rows: list[tuple[Any, ...]] = []
    for index, row in enumerate(row_list):
        missing = [column for column in column_tuple if column not in row]
        if missing:
            raise ValueError(f"Generated option row {index} is missing columns: {missing}.")
        output_rows.append(tuple(row[column] for column in column_tuple))

    return GeneratedOptions(
        columns=column_tuple,
        rows=tuple(output_rows),
        units=dict(units or {}),
        broadcast=dict(broadcast or {}),
    )
