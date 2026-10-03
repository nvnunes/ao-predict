"""Joint stratified science-coordinate offsets on regular Cartesian grids."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from astropy import units as u

from . import schema
from .sampling_base import Sampler, SamplerRequest


def _grid_geometry(setup: Mapping[str, object]) -> tuple[float, np.ndarray, np.ndarray]:
    """Infer square lattice spacing and convex support from prepared setup."""
    r = np.asarray(setup[schema.KEY_SETUP_SCI_R].to_value(u.arcsec), dtype=float)
    theta = np.asarray(setup[schema.KEY_SETUP_SCI_THETA].to_value(u.deg), dtype=float)
    if r.ndim != 1 or theta.shape != r.shape or r.size < 4:
        raise ValueError("Stratified science offsets require at least four science-grid points.")
    radians = np.deg2rad(theta)
    x = r * np.cos(radians)
    y = r * np.sin(radians)
    x_axis = np.unique(np.round(x, 6))
    y_axis = np.unique(np.round(y, 6))
    if x_axis.size < 2 or x_axis.size != y_axis.size:
        raise ValueError("Science coordinates must form a square regular Cartesian lattice.")
    x_steps = np.diff(x_axis)
    y_steps = np.diff(y_axis)
    spacing = float((x_axis[-1] - x_axis[0]) / (x_axis.size - 1))
    if (
        spacing <= 0
        or not np.allclose(x_steps, spacing, rtol=0.0, atol=1e-5)
        or not np.allclose(y_steps, spacing, rtol=0.0, atol=1e-5)
    ):
        raise ValueError("Science coordinates must have uniform Cartesian grid spacing.")
    lower_x = float(x_axis[0])
    upper_x = float(x_axis[-1])
    lower_y = float(y_axis[0])
    upper_y = float(y_axis[-1])
    x = np.where(np.isclose(x, lower_x, atol=1e-10), lower_x, x)
    x = np.where(np.isclose(x, upper_x, atol=1e-10), upper_x, x)
    y = np.where(np.isclose(y, lower_y, atol=1e-10), lower_y, y)
    y = np.where(np.isclose(y, upper_y, atol=1e-10), upper_y, y)
    if (
        not np.allclose((x - lower_x) / spacing, np.rint((x - lower_x) / spacing), rtol=0.0, atol=max(1e-8, 2e-6 / spacing))
        or not np.allclose((y - lower_y) / spacing, np.rint((y - lower_y) / spacing), rtol=0.0, atol=max(1e-8, 2e-6 / spacing))
    ):
        raise ValueError("Science coordinates do not lie on a regular Cartesian lattice.")
    points = np.column_stack((np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)))
    if np.unique(points, axis=0).shape[0] != points.shape[0]:
        raise ValueError("Science coordinates contain duplicate grid points.")
    vertices = _convex_hull_vertices(points)
    return spacing, points, _convex_hull_halfspaces(vertices)


def _convex_hull_vertices(points: np.ndarray) -> np.ndarray:
    unique_points = np.unique(np.asarray(points, dtype=np.float32), axis=0)
    if unique_points.shape[0] < 3:
        raise ValueError("Science-grid support requires at least three unique points.")

    def cross(origin: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
        first_delta = first - origin
        second_delta = second - origin
        return float(first_delta[0] * second_delta[1] - first_delta[1] * second_delta[0])

    lower: list[np.ndarray] = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper: list[np.ndarray] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float32)


def _convex_hull_halfspaces(vertices: np.ndarray) -> np.ndarray:
    equations = []
    for start, stop in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
        edge = stop - start
        normal = np.asarray([edge[1], -edge[0]], dtype=np.float32)
        normal /= np.linalg.norm(normal)
        equations.append((float(normal[0]), float(normal[1]), -float(np.dot(normal, start))))
    return np.asarray(equations, dtype=np.float32)


def _raw_axis_offsets(rng: np.random.Generator, count: int, num_points: int, spacing: float) -> np.ndarray:
    offsets = rng.random((count, num_points), dtype=np.float32)
    offsets *= np.float32(spacing)
    offsets -= np.float32(spacing / 2.0)
    return offsets


def _validate_grid_coverage(spacing: float, base: np.ndarray, halfspaces: np.ndarray) -> None:
    """Require source cells to cover the convex support without interior gaps."""
    if len(_convex_hull_vertices(base)) < 3:
        raise ValueError("Redistributed science offsets require convex grid support with positive area.")
    lower = base.min(axis=0).astype(float)
    indices = np.rint((base - lower) / spacing).astype(int)
    occupied = set(map(tuple, indices))
    half_cell = spacing / 2.0
    for x_index in range(int(indices[:, 0].max()) + 1):
        for y_index in range(int(indices[:, 1].max()) + 1):
            if (x_index, y_index) in occupied:
                continue
            center = lower + spacing * np.asarray([x_index, y_index])
            polygon = center + half_cell * np.asarray([[-1, -1], [1, -1], [1, 1], [-1, 1]])
            for a, b, c in halfspaces.astype(float):
                clipped = []
                for start, stop in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
                    start_distance = a * start[0] + b * start[1] + c
                    stop_distance = a * stop[0] + b * stop[1] + c
                    if start_distance <= 0.0:
                        clipped.append(start)
                    if (start_distance <= 0.0) != (stop_distance <= 0.0):
                        clipped.append(start + start_distance / (start_distance - stop_distance) * (stop - start))
                polygon = np.asarray(clipped)
                if len(polygon) < 3:
                    break
            if len(polygon) >= 3:
                polygon -= center
                area = abs(
                    np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
                    - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
                ) / 2.0
                if area > 1e-5 * spacing**2:
                    raise ValueError(
                        "Redistributed science offsets require grid cells to cover the convex grid support without gaps."
                    )


def _redistribute_offsets(
    dx: np.ndarray,
    dy: np.ndarray,
    base: np.ndarray,
    halfspaces: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Replace exterior draws uniformly within support, retaining interior offsets."""
    x = base[np.newaxis, :, 0] + dx
    y = base[np.newaxis, :, 1] + dy
    outside = np.zeros(dx.shape, dtype=bool)
    for a, b, c in halfspaces:
        outside |= a * x + b * y + c > 0.0
    rows, columns = np.nonzero(outside)
    lower = base.min(axis=0).astype(float)
    upper = base.max(axis=0).astype(float)
    while rows.size:
        positions = rng.uniform(lower, upper, size=(rows.size, 2))
        offsets = np.asarray(positions - base[columns], dtype=np.float32)
        positions = base[columns] + offsets
        inside = np.ones(rows.size, dtype=bool)
        for a, b, c in halfspaces:
            inside &= a * positions[:, 0] + b * positions[:, 1] + c <= 0.0
        dx[rows[inside], columns[inside]] = offsets[inside, 0]
        dy[rows[inside], columns[inside]] = offsets[inside, 1]
        rows = rows[~inside]
        columns = columns[~inside]


def _reflect_offsets(
    dx: np.ndarray,
    dy: np.ndarray,
    base: np.ndarray,
    halfspaces: np.ndarray,
    half_cell_arcsec: float,
) -> None:
    """Apply the retained float32 reflection and inward-quantization order."""
    dx += base[np.newaxis, :, 0]
    dy += base[np.newaxis, :, 1]
    tolerance = np.float32(1.0e-5)
    half_cell = np.float32(half_cell_arcsec)
    relevant_columns: list[np.ndarray] = []
    for a, b, c in halfspaces:
        center_distance = a * base[:, 0] + b * base[:, 1] + c
        outward_cell_extent = half_cell * (abs(a) + abs(b))
        relevant_columns.append(np.flatnonzero(center_distance + outward_cell_extent > tolerance))

    max_passes = max(2 * len(halfspaces), 1)
    for _ in range(max_passes):
        reflected = False
        for (a, b, c), columns in zip(halfspaces, relevant_columns, strict=True):
            for column in columns:
                signed_distance = a * dx[:, column] + b * dy[:, column] + c
                outside = signed_distance > tolerance
                if not np.any(outside):
                    continue
                distance = signed_distance[outside]
                dx[outside, column] -= 2.0 * distance * a
                dy[outside, column] -= 2.0 * distance * b
                reflected = True
        if not reflected:
            break

    for (a, b, c), columns in zip(halfspaces, relevant_columns, strict=True):
        for column in columns:
            signed_distance = a * dx[:, column] + b * dy[:, column] + c
            if np.any(signed_distance > tolerance):
                raise ValueError("Reflected science coordinates remain outside grid support.")

    dx -= base[np.newaxis, :, 0]
    dy -= base[np.newaxis, :, 1]
    quantization_margin = np.float32(8.0 * np.finfo(np.float32).eps * max(float(half_cell), 1.0))
    for _ in range(2):
        for (a, b, c), columns in zip(halfspaces, relevant_columns, strict=True):
            for column in columns:
                signed_distance = a * (base[column, 0] + dx[:, column]) + b * (base[column, 1] + dy[:, column]) + c
                outside = signed_distance > 0.0
                if not np.any(outside):
                    continue
                correction = signed_distance[outside] + quantization_margin
                dx[outside, column] -= correction * a
                dy[outside, column] -= correction * b

    for (a, b, c), columns in zip(halfspaces, relevant_columns, strict=True):
        for column in columns:
            signed_distance = a * (base[column, 0] + dx[:, column]) + b * (base[column, 1] + dy[:, column]) + c
            if np.any(signed_distance > tolerance):
                raise ValueError("Quantized science-coordinate offsets exceed grid support.")
    if np.any(np.abs(dx) > half_cell + tolerance) or np.any(np.abs(dy) > half_cell + tolerance):
        raise ValueError("Science-coordinate reflection left its source grid cell.")


def _validate_science_offset_declaration(request: SamplerRequest, name: str) -> None:
    if request.fields != (schema.KEY_OPTION_SCI_DX, schema.KEY_OPTION_SCI_DY):
        raise ValueError(f"{name} requires sci_dx owner and sci_dy: '@sci_dx'.")
    if request.parameters:
        raise ValueError(f"{name} parameters must be empty.")


class StratifiedScienceOffsetsSampler(Sampler):
    """Draw joint grid-cell offsets, reflecting exterior points within support.

    The regular grid's convex hull defines the field boundary. Reflected offsets
    remain within half a grid spacing per axis of their source point. Outputs
    are float32 arcsecond matrices with one column per science point.
    """

    version = 1

    @classmethod
    def validate_declaration(cls, request: SamplerRequest) -> None:
        _validate_science_offset_declaration(request, "stratified_science_offsets")
        _grid_geometry(request.setup)

    def sample(self, request: SamplerRequest) -> Mapping[str, np.ndarray | u.Quantity]:
        spacing, base, halfspaces = _grid_geometry(request.setup)
        rng = np.random.default_rng(request.seed)
        dx = _raw_axis_offsets(rng, request.count, base.shape[0], spacing)
        dy = _raw_axis_offsets(rng, request.count, base.shape[0], spacing)
        _reflect_offsets(dx, dy, base, halfspaces, spacing / 2.0)
        return {
            schema.KEY_OPTION_SCI_DX: dx * u.arcsec,
            schema.KEY_OPTION_SCI_DY: dy * u.arcsec,
        }


class StratifiedScienceOffsetsRedistributedSampler(Sampler):
    """Draw joint grid-cell offsets, redistributing exterior points within support.

    Source cells must cover the regular grid's convex hull without gaps.
    Interior offsets are retained; exterior points are replaced by area-uniform
    positions inside that hull. Replacement offsets may leave their source cell.
    Outputs are float32 arcsecond matrices with one column per science point.
    """

    version = 1

    @classmethod
    def validate_declaration(cls, request: SamplerRequest) -> None:
        _validate_science_offset_declaration(request, "stratified_science_offsets_redistributed")
        spacing, base, halfspaces = _grid_geometry(request.setup)
        _validate_grid_coverage(spacing, base, halfspaces)

    def sample(self, request: SamplerRequest) -> Mapping[str, np.ndarray | u.Quantity]:
        spacing, base, halfspaces = _grid_geometry(request.setup)
        _validate_grid_coverage(spacing, base, halfspaces)
        rng = np.random.default_rng(request.seed)
        dx = _raw_axis_offsets(rng, request.count, base.shape[0], spacing)
        dy = _raw_axis_offsets(rng, request.count, base.shape[0], spacing)
        _redistribute_offsets(dx, dy, base, halfspaces, rng)
        return {
            schema.KEY_OPTION_SCI_DX: dx * u.arcsec,
            schema.KEY_OPTION_SCI_DY: dy * u.arcsec,
        }
