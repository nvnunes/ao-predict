"""Seeded construction of canonical simulation-option populations.

Samplers draw declared option fields from read-only prepared simulation and
setup context. Generation returns ordinary ``OptionsConfig`` arrays; sampler
recipes are not part of the persisted dataset contract.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import importlib
from types import MappingProxyType
from typing import Mapping, TYPE_CHECKING

import numpy as np
from astropy import units as u

from . import schema
from .config import _parse_broadcast_defaults, prepare_options_payload_from_arrays
from .interfaces import Simulation
from .sampling_base import Sampler, SamplerRequest
from .sampling_offsets import StratifiedScienceOffsetsSampler

if TYPE_CHECKING:
    from .api import OptionsConfig, SetupConfig, SimulationConfig


_SEED_PREFIX = b"ao-predict:option-field-seed:v1\0"
_GENERATED_FIELDS = frozenset(
    schema.OPTION_KEYS_1D + schema.OPTION_KEYS_NGS + schema.OPTION_KEYS_SCI_OFFSETS
)


@dataclass(frozen=True)
class GenerateOptionsConfig:
    """Configuration for a reproducible option population.

    Attributes:
        count: Positive number of option rows.
        num_ngs: Positive width of generated NGS arrays.
        fields: Canonical option names mapped to sampler definitions or direct
            ``"@owner"`` references.
        seed: Integer root seed; omission uses the fixed value zero.
        broadcast: Existing scalar or NGS broadcast defaults, disjoint from
            generated fields.
    """

    count: int
    num_ngs: int
    fields: Mapping[str, object]
    seed: int | None = None
    broadcast: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerateOptionsRequest:
    """Prepare simulation and setup, then generate complete canonical options.

    Attributes:
        simulation: Existing typed or mapping simulation configuration.
        setup: Existing typed or mapping setup configuration.
        options: Generation recipe and optional broadcast defaults.
    """

    simulation: SimulationConfig | Mapping[str, object]
    setup: SetupConfig | Mapping[str, object]
    options: GenerateOptionsConfig


def _freeze(value: object) -> object:
    """Copy nested inputs into read-only sampler-facing values."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, u.Quantity):
        array = _freeze_array(np.asarray(value.value))
        return u.Quantity(array, unit=value.unit, copy=False)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return _freeze(value.tolist())
        return _freeze_array(value)
    return deepcopy(value)


def _freeze_array(value: np.ndarray) -> np.ndarray:
    """Use an immutable bytes owner so a sampler cannot re-enable writes."""
    contiguous = np.ascontiguousarray(value)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(value.shape)


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return int(value)


def _root_seed(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("options.generate.seed must be an integer, not a Boolean.")
    return int(value)


def _field_seed(root_seed: int, owner_field: str) -> int:
    payload = _SEED_PREFIX + str(root_seed).encode("ascii") + b"\0" + owner_field.encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:4], "little")


def _field_shape(field_name: str, count: int, num_ngs: int, setup: Mapping[str, object]) -> tuple[int, ...]:
    if field_name in schema.OPTION_KEYS_NGS:
        return count, num_ngs
    if field_name in schema.OPTION_KEYS_SCI_OFFSETS:
        return count, len(np.asarray(setup[schema.KEY_SETUP_SCI_R]))
    return (count,)


def _unit_for_definition(field_name: str, definition: Mapping[str, object]) -> u.UnitBase | None:
    canonical = schema.OPTION_FIELD_UNITS.get(field_name)
    unit_name = definition.get("unit")
    if canonical is None:
        if unit_name is not None:
            raise ValueError(f"options.generate.fields.{field_name}.unit is not allowed for a nonphysical field.")
        return None
    if unit_name is None:
        raise ValueError(f"options.generate.fields.{field_name}.unit is required.")
    try:
        unit = u.Unit(unit_name)
        (1 * unit).to(canonical)
    except (TypeError, ValueError, u.UnitConversionError) as exc:
        raise ValueError(f"options.generate.fields.{field_name}.unit must be compatible with {canonical}.") from exc
    return unit


def _resolve_sampler(name: str) -> type[Sampler]:
    builtin = _BUILTINS.get(name)
    if builtin is not None:
        return builtin
    if ":" in name:
        module_name, class_name = name.split(":", 1)
    else:
        parts = name.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(f"Unknown sampler '{name}'. Use a built-in name or module:ClassName.")
        module_name, class_name = parts
    if not module_name or not class_name:
        raise ValueError(f"Invalid sampler path '{name}'.")
    module = importlib.import_module(module_name)
    sampler_class = getattr(module, class_name, None)
    if not isinstance(sampler_class, type) or not issubclass(sampler_class, Sampler):
        raise ValueError(f"'{name}' does not resolve to a Sampler subclass.")
    return sampler_class


def _validate_output(request: SamplerRequest, result: object) -> dict[str, np.ndarray | u.Quantity]:
    if not isinstance(result, Mapping):
        raise ValueError(f"Sampler for '{request.owner_field}' must return a field mapping.")
    if set(result) != set(request.fields):
        raise ValueError(
            f"Sampler for '{request.owner_field}' returned fields {sorted(result)}; "
            f"expected {sorted(request.fields)}."
        )
    out: dict[str, np.ndarray | u.Quantity] = {}
    for field_name in request.fields:
        value = result[field_name]
        canonical = schema.OPTION_FIELD_UNITS.get(field_name)
        if canonical is not None:
            if not isinstance(value, u.Quantity):
                raise ValueError(f"Sampler output '{field_name}' must be an Astropy Quantity.")
            try:
                converted = value.to(canonical)
            except u.UnitConversionError as exc:
                raise ValueError(f"Sampler output '{field_name}' has an incompatible unit.") from exc
            array = np.asarray(converted.value)
            output: np.ndarray | u.Quantity = converted
        else:
            if not isinstance(value, np.ndarray):
                raise ValueError(f"Sampler output '{field_name}' must be a NumPy array.")
            array = value
            output = value
        expected = _field_shape(field_name, request.count, request.num_ngs, request.setup)
        if array.shape != expected:
            raise ValueError(f"Sampler output '{field_name}' must have shape {expected}, got {array.shape}.")
        if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
            raise ValueError(f"Sampler output '{field_name}' must contain finite numeric values.")
        out[field_name] = output
    return out


def _broadcast_arrays(config: GenerateOptionsConfig, generated: Mapping[str, object]) -> dict[str, object]:
    scalars, ngs = _parse_broadcast_defaults(config.broadcast)
    out: dict[str, object] = {}
    for name, value in scalars.items():
        if name in generated:
            raise ValueError(f"options.broadcast.{name} conflicts with options.generate.fields.{name}.")
        array = np.full((config.count,), value)
        unit = schema.OPTION_FIELD_UNITS.get(name)
        out[name] = array * unit if unit is not None else array
    if ngs:
        if set(ngs) & set(generated):
            raise ValueError("options.broadcast.ngs conflicts with generated NGS fields.")
        for name, values in ngs.items():
            if len(values) != config.num_ngs:
                raise ValueError(f"options.broadcast.{name} must have num_ngs={config.num_ngs} slots.")
            array = np.broadcast_to(values, (config.count, config.num_ngs)).copy()
            out[name] = array * schema.OPTION_FIELD_UNITS[name]
    return out


def _resolved_requests(
    config: GenerateOptionsConfig,
    simulation_payload: Mapping[str, object],
    setup_payload: Mapping[str, object],
) -> list[tuple[type[Sampler], SamplerRequest]]:
    count = _positive_int(config.count, "options.generate.count")
    num_ngs = _positive_int(config.num_ngs, "options.generate.num_ngs")
    seed = _root_seed(config.seed)
    if not isinstance(config.fields, Mapping) or not config.fields:
        raise ValueError("options.generate.fields must be a nonempty mapping.")
    definitions = dict(config.fields)
    if any(not isinstance(name, str) or name not in _GENERATED_FIELDS for name in definitions):
        raise ValueError("options.generate.fields contains an unsupported canonical option field.")
    references: dict[str, str] = {}
    owners: dict[str, Mapping[str, object]] = {}
    for name, definition in definitions.items():
        if isinstance(definition, str) and definition.startswith("@"):
            references[name] = definition[1:]
        elif isinstance(definition, Mapping):
            owners[name] = definition
        else:
            raise ValueError(f"options.generate.fields.{name} must be a sampler definition or '@owner' reference.")
    for name, owner in references.items():
        if owner not in owners:
            raise ValueError(f"options.generate.fields.{name} must refer directly to a defined owner, not '{owner}'.")
        if owner == name:
            raise ValueError(f"options.generate.fields.{name} cannot refer to itself.")
    resolved: list[tuple[type[Sampler], SamplerRequest]] = []
    for name, definition in owners.items():
        label = f"options.generate.fields.{name}"
        if set(definition) - {"sampler", "version", "parameters", "unit"}:
            raise ValueError(f"{label} contains unsupported keys.")
        sampler_name = definition.get("sampler")
        if not isinstance(sampler_name, str) or not sampler_name:
            raise ValueError(f"{label}.sampler must be a nonempty name.")
        version = _positive_int(definition.get("version"), f"{label}.version")
        parameters = definition.get("parameters")
        if not isinstance(parameters, Mapping):
            raise ValueError(f"{label}.parameters must be a mapping.")
        unit = _unit_for_definition(name, definition)
        sampler_class = _resolve_sampler(sampler_name)
        actual_version = getattr(sampler_class, "version", None)
        if isinstance(actual_version, bool) or not isinstance(actual_version, int) or actual_version <= 0:
            raise ValueError(f"Sampler '{sampler_name}' must declare a positive integer version.")
        if version != actual_version:
            raise ValueError(f"{label}.version is {version}, but sampler '{sampler_name}' is version {actual_version}.")
        fields = (name,) + tuple(field_name for field_name, owner in references.items() if owner == name)
        request = SamplerRequest(
            owner_field=name,
            fields=fields,
            count=count,
            num_ngs=num_ngs,
            seed=_field_seed(seed, name),
            parameters=_freeze(parameters),
            unit=unit,
            simulation=_freeze(simulation_payload),
            setup=_freeze(setup_payload),
        )
        sampler_class.validate_declaration(request)
        resolved.append((sampler_class, request))
    return resolved


def external_sampler_fields(config: GenerateOptionsConfig) -> frozenset[str]:
    """Return fields declared under externally resolved sampler owners."""
    owners = {
        name
        for name, definition in config.fields.items()
        if isinstance(definition, Mapping)
        and isinstance(definition.get("sampler"), str)
        and definition["sampler"] not in _BUILTINS
    }
    return frozenset(
        name
        for name, definition in config.fields.items()
        if name in owners or (isinstance(definition, str) and definition.startswith("@") and definition[1:] in owners)
    )


def prepare_generated_options(
    simulation: Simulation,
    simulation_payload: Mapping[str, object],
    setup_payload: Mapping[str, object],
    config: GenerateOptionsConfig,
) -> OptionsConfig:
    """Generate and complete options through the existing array lifecycle."""
    from .api import OptionsConfig

    requests = _resolved_requests(config, simulation_payload, setup_payload)
    generated_names = {field_name for _, request in requests for field_name in request.fields}
    broadcast = _broadcast_arrays(config, {name: None for name in generated_names})
    arrays: dict[str, object] = {}
    for sampler_class, request in requests:
        arrays.update(_validate_output(request, sampler_class().sample(request)))
    arrays.update(broadcast)
    if not arrays:
        raise ValueError("options.generate must produce at least one option field.")
    return OptionsConfig(prepare_options_payload_from_arrays(simulation, setup_payload, arrays))


def generate_options(request: GenerateOptionsRequest) -> OptionsConfig:
    """Return complete canonical option arrays without creating a dataset.

    Simulation and setup are prepared first, exactly as for dataset
    initialization. The same normalized request and sampler versions replay
    exactly within the producing environment.
    """
    from .api import _prepare_setup_payload, _prepare_simulation_payload
    from .config import normalize_setup_config, normalize_simulation_config

    simulation, simulation_payload = _prepare_simulation_payload(normalize_simulation_config(request.simulation))
    setup_payload = _prepare_setup_payload(simulation, normalize_setup_config(request.setup))
    return prepare_generated_options(simulation, simulation_payload, setup_payload, request.options)


class UniformSampler(Sampler):
    """Draw independent bounded continuous values for one canonical field."""

    version = 1

    @classmethod
    def validate_declaration(cls, request: SamplerRequest) -> None:
        if len(request.fields) != 1:
            raise ValueError("uniform sampler returns exactly one field.")
        _numeric_bounds(request, {"minimum", "maximum"})

    def sample(self, request: SamplerRequest) -> Mapping[str, np.ndarray | u.Quantity]:
        minimum, maximum = _numeric_bounds(request, {"minimum", "maximum"})
        shape = _field_shape(request.owner_field, request.count, request.num_ngs, request.setup)
        values = np.random.RandomState(request.seed).uniform(minimum, maximum, size=shape)
        return {request.owner_field: values * request.unit if request.unit is not None else values}


def _numeric_bounds(request: SamplerRequest, expected: set[str]) -> tuple[float, float]:
    if set(request.parameters) != expected:
        raise ValueError(f"{request.owner_field} sampler parameters must be {sorted(expected)}.")
    try:
        minimum = float(request.parameters["minimum"])
        maximum = float(request.parameters["maximum"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{request.owner_field} bounds must be numeric scalars.") from exc
    if not np.isfinite(minimum) or not np.isfinite(maximum) or minimum >= maximum:
        raise ValueError(f"{request.owner_field} requires finite minimum < maximum.")
    return minimum, maximum


class WeightedDiscreteSampler(Sampler):
    """Draw with replacement from caller-supplied ordered values and weights."""

    version = 1

    @classmethod
    def validate_declaration(cls, request: SamplerRequest) -> None:
        if len(request.fields) != 1:
            raise ValueError("weighted_discrete sampler returns exactly one field.")
        _weighted_input(request)

    def sample(self, request: SamplerRequest) -> Mapping[str, np.ndarray | u.Quantity]:
        values, probabilities = _weighted_input(request)
        shape = _field_shape(request.owner_field, request.count, request.num_ngs, request.setup)
        draws = np.random.RandomState(request.seed).choice(values, size=shape, replace=True, p=probabilities)
        return {request.owner_field: draws * request.unit if request.unit is not None else draws}


def _weighted_input(request: SamplerRequest) -> tuple[np.ndarray, np.ndarray]:
    if set(request.parameters) != {"values", "weights"}:
        raise ValueError(f"{request.owner_field} weighted_discrete requires values and weights.")
    try:
        values = np.asarray(request.parameters["values"], dtype=float)
        weights = np.asarray(request.parameters["weights"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{request.owner_field} values and weights must be numeric arrays.") from exc
    if values.ndim != 1 or values.size == 0 or weights.shape != values.shape:
        raise ValueError(f"{request.owner_field} values and weights must be nonempty equal-length vectors.")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError(f"{request.owner_field} values and weights must be finite, with nonnegative weights.")
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"{request.owner_field} weights must have a positive finite sum.")
    return values, weights / total


class UniformMeanMagnitudeSampler(Sampler):
    """Draw exchangeable one-, two-, or three-NGS tuples with uniform means."""

    version = 1

    @classmethod
    def validate_declaration(cls, request: SamplerRequest) -> None:
        if request.fields != (schema.KEY_OPTION_NGS_MAGNITUDE,) or request.num_ngs not in (1, 2, 3):
            raise ValueError("uniform_mean_mag requires ngs_magnitude and num_ngs of 1, 2, or 3.")
        _numeric_bounds(request, {"minimum", "maximum"})

    def sample(self, request: SamplerRequest) -> Mapping[str, np.ndarray | u.Quantity]:
        minimum, maximum = _numeric_bounds(request, {"minimum", "maximum"})
        rng = np.random.RandomState(request.seed)
        if request.num_ngs == 1:
            values = rng.uniform(minimum, maximum, size=(request.count, 1))
        else:
            values = np.stack(
                [_uniform_mean_tuple(minimum, maximum, request.num_ngs, rng) for _ in range(request.count)]
            )
        return {schema.KEY_OPTION_NGS_MAGNITUDE: values * request.unit}


def _uniform_mean_tuple(minimum: float, maximum: float, num_ngs: int, rng: np.random.RandomState) -> np.ndarray:
    """Retain the fixed-arity mean draw and permutation order."""
    target_mean = float(rng.uniform(minimum, maximum))
    if num_ngs == 2:
        half_difference_max = min(target_mean - minimum, maximum - target_mean)
        half_difference = float(rng.uniform(0.0, half_difference_max))
        magnitudes = np.asarray([target_mean - half_difference, target_mean + half_difference])
    else:
        magnitudes = _three_magnitudes_with_mean(target_mean, minimum, maximum, rng)
    return magnitudes[rng.permutation(num_ngs)]


def _three_magnitudes_with_mean(
    target_mean: float, minimum: float, maximum: float, rng: np.random.RandomState
) -> np.ndarray:
    span = maximum - minimum
    total = 3.0 * (target_mean - minimum)
    if total <= np.finfo(float).eps * span:
        return np.full(3, minimum)
    if 3.0 * span - total <= np.finfo(float).eps * span:
        return np.full(3, maximum)
    polygon = np.asarray(((0.0, 0.0), (span, 0.0), (span, span), (0.0, span)), dtype=float)
    polygon = _clip_sum_polygon(polygon, total, keep_below=True)
    polygon = _clip_sum_polygon(polygon, total - span, keep_below=False)
    point = _sample_uniform_polygon(polygon, rng)
    offsets = np.asarray((point[0], point[1], total - point.sum()))
    return np.clip(minimum + offsets, minimum, maximum)


def _clip_sum_polygon(polygon: np.ndarray, boundary: float, *, keep_below: bool) -> np.ndarray:
    if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
        raise ValueError("A polygon must contain at least three 2D vertices.")

    def signed_distance(point: np.ndarray) -> float:
        value = float(point.sum() - boundary)
        return value if keep_below else -value

    clipped: list[np.ndarray] = []
    previous = polygon[-1]
    previous_distance = signed_distance(previous)
    for current in polygon:
        current_distance = signed_distance(current)
        previous_inside = previous_distance <= 0.0
        current_inside = current_distance <= 0.0
        if previous_inside != current_inside:
            fraction = previous_distance / (previous_distance - current_distance)
            clipped.append(previous + fraction * (current - previous))
        if current_inside:
            clipped.append(current)
        previous = current
        previous_distance = current_distance
    result = np.asarray(clipped, dtype=float)
    if result.ndim != 2 or result.shape[0] < 3:
        raise ValueError("The feasible constant-mean magnitude polygon is empty.")
    return result


def _sample_uniform_polygon(polygon: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    origin = polygon[0]
    first = polygon[1:-1] - origin
    second = polygon[2:] - origin
    twice_areas = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    total_area = float(twice_areas.sum())
    if not total_area > 0.0:
        raise ValueError("The feasible constant-mean magnitude polygon has zero area.")
    triangle = int(np.searchsorted(np.cumsum(twice_areas), rng.uniform(0.0, total_area), side="right"))
    first_weight, second_weight = rng.uniform(size=2)
    if first_weight + second_weight > 1.0:
        first_weight = 1.0 - first_weight
        second_weight = 1.0 - second_weight
    return origin + first_weight * (polygon[triangle + 1] - origin) + second_weight * (polygon[triangle + 2] - origin)


class UniformAreaRadiusSampler(Sampler):
    """Draw polar radii with constant area density inside a supplied disk."""

    version = 1

    @classmethod
    def validate_declaration(cls, request: SamplerRequest) -> None:
        if request.fields != (schema.KEY_OPTION_NGS_R,):
            raise ValueError("uniform_area_radius requires the ngs_r field.")
        _positive_maximum(request)

    def sample(self, request: SamplerRequest) -> Mapping[str, np.ndarray | u.Quantity]:
        maximum = _positive_maximum(request)
        values = maximum * np.sqrt(np.random.RandomState(request.seed).random((request.count, request.num_ngs)))
        return {schema.KEY_OPTION_NGS_R: values * request.unit}


def _positive_maximum(request: SamplerRequest) -> float:
    if set(request.parameters) != {"maximum"}:
        raise ValueError(f"{request.owner_field} requires only a maximum parameter.")
    try:
        maximum = float(request.parameters["maximum"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{request.owner_field} maximum must be a positive scalar.") from exc
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError(f"{request.owner_field} maximum must be finite and positive.")
    return maximum


class UniformThetaSampler(Sampler):
    """Draw a full-turn polar angle using the retained radians formula."""

    version = 1

    @classmethod
    def validate_declaration(cls, request: SamplerRequest) -> None:
        if request.fields != (schema.KEY_OPTION_NGS_THETA,) or request.parameters:
            raise ValueError("uniform_theta requires ngs_theta and empty parameters.")

    def sample(self, request: SamplerRequest) -> Mapping[str, np.ndarray | u.Quantity]:
        radians = np.random.RandomState(request.seed).uniform(
            0.0, 2.0 * np.pi, (request.count, request.num_ngs)
        )
        return {schema.KEY_OPTION_NGS_THETA: (radians * u.rad).to(request.unit)}


_BUILTINS: dict[str, type[Sampler]] = {
    "uniform": UniformSampler,
    "weighted_discrete": WeightedDiscreteSampler,
    "uniform_mean_mag": UniformMeanMagnitudeSampler,
    "uniform_area_radius": UniformAreaRadiusSampler,
    "uniform_theta": UniformThetaSampler,
    "stratified_science_offsets": StratifiedScienceOffsetsSampler,
}
