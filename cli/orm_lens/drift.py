"""Schema drift — do the migrations still describe the models?

Django answers this with ``makemigrations --check``, which needs a working
settings module, an importable app registry, and every third-party dependency
installed. On a cold clone, a broken venv, or a CI job that deliberately does
not boot the project, that answer is unavailable exactly when it would be
cheapest to act on.

This module answers it from source. It replays each app's migrations in order
— ``CreateModel``, ``AddField``, ``RemoveField``, ``RenameField``,
``DeleteModel``, ``RenameModel`` — into the set of fields the migrations
believe each model has, then compares that against what ``models.py`` actually
declares.

Two kinds of drift come out, and they fail in opposite directions:

* **A field in models.py with no migration.** Someone edited a model and
  forgot ``makemigrations``. Deploy and the column is missing under a model
  that expects it — an error at runtime, on the first query that touches it.
* **A field in the migrations that no model declares.** The column exists and
  nothing reads it. Usually harmless, occasionally a half-finished removal
  still holding a ``NOT NULL`` constraint.

Being static, this cannot see what Django computes at import time — abstract
bases resolved through metaclasses, fields injected by third-party mixins,
swappable models. Those surface as drift that is not real, which is why the
report separates the two directions and never blocks on the "extra in
migrations" side alone.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .migrations_parser import (
    _build_current_schema,
    _find_all_migration_apps,
    _find_migration_class,
    _iter_migration_files,
    _literal_or_none,
    _replay_order,
)

# Operations that change which fields a model has. AlterField is absent on
# purpose: it changes a column's type or options, never the field's existence,
# so it cannot produce drift in the sense measured here.
_STRUCTURAL_OPS = frozenset(
    {
        "CreateModel",
        "DeleteModel",
        "RenameModel",
        "AddField",
        "RemoveField",
        "RenameField",
    }
)

# Django adds these itself unless told otherwise, so a migration that omits
# them is not drift.
_IMPLICIT_FIELDS = frozenset({"id"})


@dataclass
class ModelDrift:
    """One model whose migrations and declaration disagree."""

    app: str
    model: str
    missing_in_migrations: list[str] = field(default_factory=list)
    """Declared in models.py, never migrated — the dangerous direction."""
    missing_in_models: list[str] = field(default_factory=list)
    """Migrated, but no longer declared."""
    only_in_models: bool = False
    """The model has no CreateModel anywhere."""
    only_in_migrations: bool = False
    """The migrations create it; models.py does not declare it."""

    @property
    def label(self) -> str:
        return f"{self.app}.{self.model}"

    @property
    def is_blocking(self) -> bool:
        """Only the "code expects a column that will not exist" direction.

        Extra columns are noise far more often than they are bugs, and static
        analysis cannot see mixin-injected fields — failing a build on those
        would train people to pass ``--exit-zero`` forever.
        """
        return bool(self.missing_in_migrations) or self.only_in_models

    def to_dict(self) -> dict[str, Any]:
        return {
            "app": self.app,
            "model": self.model,
            "missingInMigrations": sorted(self.missing_in_migrations),
            "missingInModels": sorted(self.missing_in_models),
            "onlyInModels": self.only_in_models,
            "onlyInMigrations": self.only_in_migrations,
            "blocking": self.is_blocking,
        }


@dataclass
class DriftReport:
    drifted: list[ModelDrift] = field(default_factory=list)
    apps_scanned: int = 0
    migrations_replayed: int = 0

    @property
    def blocking_count(self) -> int:
        return sum(1 for d in self.drifted if d.is_blocking)

    def to_dict(self) -> dict[str, Any]:
        return {
            "drifted": [d.to_dict() for d in self.drifted],
            "summary": {
                "apps": self.apps_scanned,
                "migrations": self.migrations_replayed,
                "drifted": len(self.drifted),
                "blocking": self.blocking_count,
            },
        }


def _op_name(call: ast.Call) -> str | None:
    """``migrations.AddField(...)`` → ``AddField``."""
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _kwarg(call: ast.Call, name: str) -> Any:
    for kw in call.keywords:
        if kw.arg == name:
            return _literal_or_none(kw.value)
    return None


def _create_model_fields(call: ast.Call) -> list[str]:
    """Field names out of ``CreateModel(fields=[("title", ...), ...])``.

    The second element of each pair is a field *constructor call*, which
    ``ast.literal_eval`` cannot touch — so the names are read positionally
    rather than by evaluating the list.
    """
    names: list[str] = []
    for kw in call.keywords:
        if kw.arg != "fields":
            continue
        if not isinstance(kw.value, (ast.List, ast.Tuple)):
            continue
        for item in kw.value.elts:
            if isinstance(item, (ast.Tuple, ast.List)) and item.elts:
                first = item.elts[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    names.append(first.value)
    return names


@dataclass
class _ListBinding:
    """A ``NAME = [...]`` value and the bindings it could see when built.

    A list is built once, so names inside it resolve against the bindings
    made before the assignment — never whatever the body later rebinds them
    to.
    """

    value: ast.expr
    env: dict[str, _ListBinding]


def _body_lists(
    body: list[ast.stmt],
    before: ast.stmt | None = None,
    base: dict[str, _ListBinding] | None = None,
) -> dict[str, _ListBinding]:
    """Resolvable ``NAME = [...]`` bindings visible in a body, by name.

    With ``before``, only the bindings made before that statement. Python
    evaluates the right-hand side where it is written, so ``state_operations=
    STATE`` keeps the list ``STATE`` named at that point: a later ``STATE = []``
    (bound to keep the list away from the next migration) is not what the
    wrapper holds, and must not decide what the wrapper replays.

    Each list keeps the bindings it was built with, so a nested wrapper
    resolves the names it captured even when they are rebound afterwards. A
    later assignment to a name wins even when its value cannot be resolved,
    and shadows an enclosing binding of the same spelling rather than falling
    back to it. ``base`` seeds the bindings visible from the enclosing scope.
    """
    lists: dict[str, _ListBinding] = dict(base or {})
    for node in body:
        if node is before:
            break
        if isinstance(node, ast.Assign):
            value = node.value
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            value = node.value
            names = [node.target.id] if isinstance(node.target, ast.Name) else []
        else:
            continue
        binding: _ListBinding | None
        if isinstance(value, (ast.List, ast.Tuple)):
            binding = _ListBinding(value, dict(lists))
        elif isinstance(value, ast.Name) and value.id in lists:
            # ``STATE = EMPTY`` keeps the same list object, and that list
            # resolved its own names when it was built.
            binding = lists[value.id]
        else:
            # The name now holds something we cannot resolve. The earlier
            # list is still reachable through a binding that captured it.
            binding = None
        for name in names:
            if binding is None:
                lists.pop(name, None)
            else:
                lists[name] = binding
    return lists


def _module_level_lists(tree: ast.AST) -> dict[str, _ListBinding]:
    """Module-scope ``NAME = [...]`` bindings, by name.

    Hand-written migrations sometimes hoist a shared operations list into a
    constant and pass it as ``state_operations=NAME``. Those are statically
    determinable, so resolve them instead of dropping the wrapper contents.
    """
    return _body_lists(getattr(tree, "body", []))


def _state_operations(
    elts: list[ast.expr],
    constants: dict[str, _ListBinding],
    seen: frozenset[int] = frozenset(),
) -> list[ast.Call]:
    """Flatten ``SeparateDatabaseAndState`` into the operations it wraps.

    Django builds project state from ``state_operations`` only; the wrapper
    is the standard way to move a model between apps or to change state
    without touching the table. ``database_operations`` is SQL-only and must
    not seed fields.

    A literal list, a tuple, or a name bound to one is replayed; anything else
    (a call, a comprehension, a name we cannot resolve) is skipped, like every
    other construct this parser cannot evaluate. A name resolves against the
    bindings its list saw when it was built, and ``seen`` stops a binding that
    somehow points back into itself instead of recursing forever.
    """
    out: list[ast.Call] = []
    for elt in elts:
        if not isinstance(elt, ast.Call):
            continue
        if _op_name(elt) != "SeparateDatabaseAndState":
            out.append(elt)
            continue
        # Signature: (database_operations, state_operations) — real
        # projects use the keyword form, but the positional one is cheap.
        state_node = elt.args[1] if len(elt.args) > 1 else None
        for kw in elt.keywords:
            if kw.arg == "state_operations":
                state_node = kw.value
        env = constants
        if isinstance(state_node, ast.Name):
            binding = constants.get(state_node.id)
            if binding is None:
                continue
            state_node, env = binding.value, binding.env
        if isinstance(state_node, (ast.List, ast.Tuple)) and id(state_node) not in seen:
            out.extend(_state_operations(state_node.elts, env, seen | {id(state_node)}))
    return out


def _operations(tree: ast.AST) -> list[ast.Call]:
    """Every call inside a ``Migration.operations = [...]`` list."""
    out: list[ast.Call] = []
    module_constants = _module_level_lists(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "operations" not in targets:
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            continue
        # Django's own docs put shared lists on the ``Migration`` class, where
        # the name is only bound inside that class body — so class bindings
        # shadow (and extend) the module-level ones for this assignment. Only
        # the ones made before it: a name bound again below is a different
        # object by then, and the wrapper keeps the one it was given.
        constants = module_constants
        for cls in ast.walk(tree):
            if isinstance(cls, ast.ClassDef) and any(stmt is node for stmt in cls.body):
                constants = _body_lists(cls.body, before=node, base=module_constants)
                break
        out.extend(_state_operations(node.value.elts, constants))
    return out


def _apply(state: dict[str, set[str]], op: str, call: ast.Call) -> None:
    if op == "CreateModel":
        name = _kwarg(call, "name")
        if isinstance(name, str):
            state[name.lower()] = set(_create_model_fields(call)) - _IMPLICIT_FIELDS
        return
    if op == "DeleteModel":
        name = _kwarg(call, "name")
        if isinstance(name, str):
            state.pop(name.lower(), None)
        return
    if op == "RenameModel":
        old, new = _kwarg(call, "old_name"), _kwarg(call, "new_name")
        if isinstance(old, str) and isinstance(new, str):
            state[new.lower()] = state.pop(old.lower(), set())
        return

    model = _kwarg(call, "model_name")
    if not isinstance(model, str):
        return
    fields = state.setdefault(model.lower(), set())
    if op == "AddField":
        name = _kwarg(call, "name")
        if isinstance(name, str):
            fields.add(name)
    elif op == "RemoveField":
        name = _kwarg(call, "name")
        if isinstance(name, str):
            fields.discard(name)
    elif op == "RenameField":
        old, new = _kwarg(call, "old_name"), _kwarg(call, "new_name")
        if isinstance(old, str) and isinstance(new, str):
            fields.discard(old)
            fields.add(new)


def replay_app(migrations_dir: Path) -> tuple[dict[str, set[str]], int]:
    """Replay one app's migrations into ``{model_lower: {field_names}}``.

    :returns: the state, and how many migration files contributed to it.
    """
    state: dict[str, set[str]] = {}
    parsed: list[tuple[Path, ast.Module]] = []
    for path in _iter_migration_files(migrations_dir):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            # A migration we cannot parse is skipped rather than fatal: one
            # bad file must not blank out the whole app's state.
            continue
        if _find_migration_class(tree) is None:
            # Helper modules (constants, mixins) match the filename filter but
            # are not migrations, so they contribute no operations.
            continue
        parsed.append((path, tree))
    replayed = len(parsed)
    # The directory name is the app's package, which is its label unless
    # `AppConfig.label` overrides it; `_replay_order` checks the dependency
    # tuples before trusting it.
    for _path, tree in _replay_order(parsed, app_label=migrations_dir.parent.name):
        for call in _operations(tree):
            op = _op_name(call)
            if op in _STRUCTURAL_OPS:
                _apply(state, op, call)
    return state, replayed


def detect_drift(
    project_root: str | Path,
    *,
    declared: dict[str, dict[str, set]] | None = None,
) -> DriftReport:
    """Compare replayed migration state against declared models.

    :param declared: pre-built ``{app: {model_lower: {fields}}}``; mostly for
        tests. Built from the workspace parser when omitted.
    """
    root = str(Path(project_root).resolve())
    if declared is None:
        declared = _build_current_schema(root)

    report = DriftReport()
    seen_apps: set[str] = set()

    # Several directories can carry the same app name — django-guardian ships
    # example_project/core and example_project_custom_group/core, and any
    # monorepo has the same shape. The declared side is already merged by name
    # (the parser labels an app by its directory), so replaying each migrations
    # directory separately compared one project's migrations against both
    # projects' models: core.customuser was printed twice, and core.customgroup
    # came out as "no migration creates it" *and* "migrated but no longer
    # declared" at once. Merge the replayed state per label so both sides agree.
    merged: dict[str, dict[str, set[str]]] = {}
    for app_label, migrations_dir in _find_all_migration_apps(root):
        migrated, replayed = replay_app(migrations_dir)
        report.migrations_replayed += replayed
        into = merged.setdefault(app_label, {})
        for model, fields in migrated.items():
            into.setdefault(model, set()).update(fields)

    for app_label, migrated in merged.items():
        seen_apps.add(app_label)
        declared_app = declared.get(app_label, {})

        for model, migrated_fields in migrated.items():
            declared_fields = declared_app.get(model)
            if declared_fields is None:
                report.drifted.append(
                    ModelDrift(app=app_label, model=model, only_in_migrations=True)
                )
                continue
            declared_set = set(declared_fields) - _IMPLICIT_FIELDS
            missing_in_migrations = sorted(declared_set - migrated_fields)
            missing_in_models = sorted(migrated_fields - declared_set)
            if missing_in_migrations or missing_in_models:
                report.drifted.append(
                    ModelDrift(
                        app=app_label,
                        model=model,
                        missing_in_migrations=missing_in_migrations,
                        missing_in_models=missing_in_models,
                    )
                )

        for model in declared_app:
            if model not in migrated:
                report.drifted.append(
                    ModelDrift(app=app_label, model=model, only_in_models=True)
                )

    report.apps_scanned = len(seen_apps)
    report.drifted.sort(key=lambda d: (not d.is_blocking, d.label))
    return report


BLOCKING_MARK = "!!"
ADVISORY_MARK = " ~"

#: Reported in #57: the marks were only discoverable by reading the source.
#: Kept next to the marks themselves so the two cannot drift apart, and
#: asserted against them in the tests.
LEGEND_LINES = (
    f"{BLOCKING_MARK} blocking — models declare something no migration creates;"
    " it will not exist at runtime",
    f"{ADVISORY_MARK} advisory — migrations keep something models no longer"
    " declare; harmless at runtime",
    "",
)


def format_drift(report: DriftReport) -> str:
    """Human-readable text. JSON is serialised by the caller."""
    if not report.drifted:
        return (
            f"no drift — {report.migrations_replayed} migration(s) across "
            f"{report.apps_scanned} app(s) match the declared models."
        )
    out: list[str] = list(LEGEND_LINES)
    for d in report.drifted:
        mark = BLOCKING_MARK if d.is_blocking else ADVISORY_MARK
        if d.only_in_models:
            out.append(f"{mark} {d.label}: declared, but no migration creates it")
            out.append("     run makemigrations, or the table will not exist")
            continue
        if d.only_in_migrations:
            out.append(
                f"{mark} {d.label}: migrated, but models.py does not declare it"
            )
            out.append("     stale table, or the model lives outside the parsed tree")
            continue
        out.append(f"{mark} {d.label}:")
        if d.missing_in_migrations:
            out.append(
                "     declared but never migrated: "
                + ", ".join(d.missing_in_migrations)
            )
            out.append("       the column will be missing at runtime")
        if d.missing_in_models:
            out.append(
                "     migrated but no longer declared: "
                + ", ".join(d.missing_in_models)
            )
    out.append("")
    out.append(
        f"summary: {len(report.drifted)} model(s) drifted, "
        f"{report.blocking_count} blocking, "
        f"{report.migrations_replayed} migration(s) replayed"
    )
    return "\n".join(out)
