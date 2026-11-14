"""Tests for static schema-drift detection (orm_lens/drift.py).

The replay has to be right in both directions, and the *blocking* decision has
to stay narrow: only "code expects a column that will not exist" fails a
build. Extra columns are noise far more often than bugs, and a static analyzer
cannot see mixin-injected fields — blocking on those would train people to
pass --exit-zero permanently, which costs more than the check is worth.

Run: python -m unittest discover cli/tests -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from orm_lens.drift import (
    ADVISORY_MARK,
    BLOCKING_MARK,
    LEGEND_LINES,
    DriftReport,
    ModelDrift,
    detect_drift,
    format_drift,
    replay_app,
)

HEADER = "from django.db import migrations, models\n\n\n"


def _migration(body: str, deps: str = "[]") -> str:
    return (
        f"{HEADER}class Migration(migrations.Migration):\n"
        f"    dependencies = {deps}\n"
        f"    operations = [\n{body}    ]\n"
    )


class ReplayTest(unittest.TestCase):
    def setUp(self) -> None:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.root = Path(td.name)
        self.migrations = self.root / "blog" / "migrations"
        self.migrations.mkdir(parents=True)

    def _write(self, name: str, body: str, deps: str = "[]") -> None:
        (self.migrations / name).write_text(_migration(body, deps), encoding="utf-8")

    def test_create_model_seeds_the_field_set(self) -> None:
        self._write(
            "0001_initial.py",
            '        migrations.CreateModel(name="Post", fields=[\n'
            '            ("id", models.AutoField(primary_key=True)),\n'
            '            ("title", models.CharField(max_length=10)),\n'
            "        ]),\n",
        )
        state, replayed = replay_app(self.migrations)
        self.assertEqual(replayed, 1)
        # `id` is implicit and must not count as drift either way.
        self.assertEqual(state, {"post": {"title"}})

    def test_add_and_remove_field(self) -> None:
        self._write(
            "0001_initial.py",
            '        migrations.CreateModel(name="Post", fields=[("title", None)]),\n',
        )
        self._write(
            "0002_x.py",
            '        migrations.AddField(model_name="post", name="slug",'
            " field=models.CharField(max_length=1)),\n"
            '        migrations.RemoveField(model_name="post", name="title"),\n',
        )
        state, _ = replay_app(self.migrations)
        self.assertEqual(state, {"post": {"slug"}})

    def test_rename_field_moves_the_name(self) -> None:
        self._write(
            "0001_initial.py",
            '        migrations.CreateModel(name="Post", fields=[("title", None)]),\n',
        )
        self._write(
            "0002_x.py",
            '        migrations.RenameField(model_name="post",'
            ' old_name="title", new_name="headline"),\n',
        )
        state, _ = replay_app(self.migrations)
        self.assertEqual(state, {"post": {"headline"}})

    def test_rename_model_carries_the_fields(self) -> None:
        self._write(
            "0001_initial.py",
            '        migrations.CreateModel(name="Post", fields=[("title", None)]),\n',
        )
        self._write(
            "0002_x.py",
            '        migrations.RenameModel(old_name="Post", new_name="Article"),\n',
        )
        state, _ = replay_app(self.migrations)
        self.assertEqual(state, {"article": {"title"}})

    def test_state_operations_inside_a_wrapper_are_replayed(self) -> None:
        # Django builds project state from `state_operations`; the wrapper is
        # how a model moves between apps without touching the table, so what
        # it contains has to count as migrated.
        self._write(
            "0001_initial.py",
            "        migrations.SeparateDatabaseAndState(\n"
            "            state_operations=[\n"
            '                migrations.CreateModel(name="Entry", fields=[\n'
            '                    ("action", None), ("message", None),\n'
            "                ]),\n"
            "            ],\n"
            "            database_operations=[],\n"
            "        ),\n",
        )
        state, _ = replay_app(self.migrations)
        self.assertEqual(state, {"entry": {"action", "message"}})

    def test_state_operations_are_read_positionally(self) -> None:
        # SeparateDatabaseAndState(database_operations, state_operations).
        self._write(
            "0001_initial.py",
            "        migrations.SeparateDatabaseAndState(\n"
            "            [],\n"
            "            [migrations.CreateModel(name='Entry', fields=[('action', None)])],\n"
            "        ),\n",
        )
        state, _ = replay_app(self.migrations)
        self.assertEqual(state, {"entry": {"action"}})

    def test_database_operations_do_not_change_state(self) -> None:
        # `database_operations` are SQL-only; Django's state never sees them.
        self._write(
            "0001_initial.py",
            '        migrations.CreateModel(name="Post", fields=[("title", None)]),\n',
        )
        self._write(
            "0002_x.py",
            "        migrations.SeparateDatabaseAndState(\n"
            "            state_operations=[],\n"
            "            database_operations=[\n"
            '                migrations.RenameModel(old_name="Post", new_name="Article"),\n'
            "            ],\n"
            "        ),\n",
        )
        state, _ = replay_app(self.migrations)
        self.assertEqual(state, {"post": {"title"}})

    def test_state_operations_bound_to_a_module_level_name_are_replayed(self) -> None:
        # Hand-written migrations hoist a shared list into a constant; the
        # wrapper still has to be followed through the name.
        (self.migrations / "0001_initial.py").write_text(
            HEADER
            + "STATE_OPERATIONS = [\n"
            + "    migrations.CreateModel(name='Entry', fields=[('action', None)]),\n"
            + "]\n\n\n"
            + "class Migration(migrations.Migration):\n"
            + "    dependencies = []\n"
            + "    operations = [\n"
            + "        migrations.SeparateDatabaseAndState(\n"
            + "            state_operations=STATE_OPERATIONS,\n"
            + "        ),\n"
            + "    ]\n",
            encoding="utf-8",
        )
        state, _ = replay_app(self.migrations)
        self.assertEqual(state, {"entry": {"action"}})

    def test_state_operations_bound_to_a_class_level_name_are_replayed(self) -> None:
        # Django's own docs put the shared list on the Migration class rather
        # than at module level, so the name is only bound inside the class body.
        (self.migrations / "0001_initial.py").write_text(
            HEADER
            + "class Migration(migrations.Migration):\n"
            + "    dependencies = []\n"
            + "    state_operations = [\n"
            + "        migrations.CreateModel(name='Entry', fields=[('action', None)]),\n"
            + "    ]\n"
            + "    operations = [\n"
            + "        migrations.SeparateDatabaseAndState(\n"
            + "            state_operations=state_operations,\n"
            + "        ),\n"
            + "    ]\n",
            encoding="utf-8",
        )
        state, _ = replay_app(self.migrations)
        self.assertEqual(state, {"entry": {"action"}})

    def test_delete_model_drops_it(self) -> None:
        self._write(
            "0001_initial.py",
            '        migrations.CreateModel(name="Post", fields=[("title", None)]),\n',
        )
        self._write("0002_x.py", '        migrations.DeleteModel(name="Post"),\n')
        self.assertEqual(replay_app(self.migrations)[0], {})

    def test_alter_field_changes_nothing_structural(self) -> None:
        self._write(
            "0001_initial.py",
            '        migrations.CreateModel(name="Post", fields=[("title", None)]),\n',
        )
        self._write(
            "0002_x.py",
            '        migrations.AlterField(model_name="post", name="title",'
            " field=models.TextField()),\n",
        )
        self.assertEqual(replay_app(self.migrations)[0], {"post": {"title"}})

    def test_migrations_apply_in_numeric_order(self) -> None:
        # 0010 must land after 0002, which string sorting would get wrong.
        self._write(
            "0001_initial.py",
            '        migrations.CreateModel(name="Post", fields=[("title", None)]),\n',
        )
        self._write(
            "0002_add.py",
            '        migrations.AddField(model_name="post", name="slug",'
            " field=None),\n",
        )
        self._write(
            "0010_drop.py",
            '        migrations.RemoveField(model_name="post", name="slug"),\n',
        )
        self.assertEqual(replay_app(self.migrations)[0], {"post": {"title"}})

    def test_hand_renamed_migration_keeps_its_place(self) -> None:
        # Django loads any module in the package, so `0001b_` is a real
        # migration. It has to replay *after* `0001_initial`: sorting it as -1
        # puts it first and the CreateModel that follows resets the field set.
        self._write(
            "0001_initial.py",
            '        migrations.CreateModel(name="CachedFile", fields=['
            '("filename", None)]),\n',
        )
        self._write(
            "0001b_add_session_key.py",
            '        migrations.AddField(model_name="cachedfile",'
            ' name="session_key", field=models.TextField(null=True)),\n',
            deps="[('blog', '0001_initial')]",
        )
        state, replayed = replay_app(self.migrations)
        self.assertEqual(replayed, 2)
        self.assertEqual(state, {"cachedfile": {"filename", "session_key"}})

    def test_dependency_only_name_replays_after_its_parent(self) -> None:
        # `add_slug.py` has no numeric prefix at all — Django applies it where
        # its `dependencies` say, which is after `0001_initial`. Applying it
        # first added the field to an empty state that CreateModel then reset.
        self._write(
            "0001_initial.py",
            '        migrations.CreateModel(name="Post", fields=[("title", None)]),\n',
        )
        self._write(
            "add_slug.py",
            '        migrations.AddField(model_name="post", name="slug",'
            " field=models.SlugField()),\n",
            deps="[('blog', '0001_initial')]",
        )
        state, replayed = replay_app(self.migrations)
        self.assertEqual(replayed, 2)
        self.assertEqual(state, {"post": {"title", "slug"}})

    def test_cross_app_dependency_is_not_a_local_edge(self) -> None:
        # `0001_initial` depends on `other/add_slug`, an app that is not this
        # one; `add_slug.py` here merely shares the name. Matching on the name
        # alone turned that into a local edge and, with `add_slug` depending on
        # `0001_initial` back, a cycle whose fallback put `add_slug` first —
        # where the later CreateModel reset its field.
        self._write(
            "0001_initial.py",
            '        migrations.CreateModel(name="Post", fields=[("title", None)]),\n',
            deps="[('other', 'add_slug')]",
        )
        self._write(
            "add_slug.py",
            '        migrations.AddField(model_name="post", name="slug",'
            " field=models.SlugField()),\n",
            deps="[('blog', '0001_initial')]",
        )
        state, replayed = replay_app(self.migrations)
        self.assertEqual(replayed, 2)
        self.assertEqual(state, {"post": {"title", "slug"}})

    def test_helper_module_without_migration_class_is_not_replayed(self) -> None:
        # `MIGRATION_FILENAME_RE` now accepts any module Django would import,
        # which also picks up constants/mixins. They are not migrations and
        # must not count towards `replayed`.
        self._write(
            "0001_initial.py",
            '        migrations.CreateModel(name="Post", fields=[("title", None)]),\n',
        )
        (self.migrations / "constants.py").write_text(
            "FIELDS = ('title',)\n", encoding="utf-8"
        )
        state, replayed = replay_app(self.migrations)
        self.assertEqual(replayed, 1)
        self.assertEqual(state, {"post": {"title"}})

    def test_unparseable_migration_is_skipped_not_fatal(self) -> None:
        self._write(
            "0001_initial.py",
            '        migrations.CreateModel(name="Post", fields=[("title", None)]),\n',
        )
        (self.migrations / "0002_broken.py").write_text("def (((", encoding="utf-8")
        state, replayed = replay_app(self.migrations)
        # One bad file must not blank out the app's state.
        self.assertEqual(state, {"post": {"title"}})
        self.assertEqual(replayed, 1)


class DetectDriftTest(unittest.TestCase):
    def setUp(self) -> None:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.root = Path(td.name)
        migrations = self.root / "blog" / "migrations"
        migrations.mkdir(parents=True)
        (migrations / "0001_initial.py").write_text(
            _migration(
                '        migrations.CreateModel(name="Post", fields=[\n'
                '            ("id", None), ("title", None),\n'
                "        ]),\n"
            ),
            encoding="utf-8",
        )

    def _detect(self, declared: dict) -> DriftReport:
        return detect_drift(self.root, declared=declared)

    def test_model_created_inside_a_wrapper_reports_no_drift(self) -> None:
        # A model moved between apps (or created without touching the table)
        # lives in `state_operations`; drift must see it like Django does.
        audit = self.root / "audit" / "migrations"
        audit.mkdir(parents=True)
        (audit / "0001_initial.py").write_text(
            _migration(
                "        migrations.SeparateDatabaseAndState(\n"
                "            state_operations=[\n"
                '                migrations.CreateModel(name="Entry", fields=[\n'
                '                    ("id", None), ("action", None), ("message", None),\n'
                "                ]),\n"
                "            ],\n"
                "        ),\n"
            ),
            encoding="utf-8",
        )
        report = self._detect(
            {
                "blog": {"post": {"id", "title"}},
                "audit": {"entry": {"id", "action", "message"}},
            }
        )
        self.assertEqual(report.drifted, [])

    def test_matching_schema_reports_no_drift(self) -> None:
        report = self._detect({"blog": {"post": {"id", "title"}}})
        self.assertEqual(report.drifted, [])
        self.assertEqual(report.blocking_count, 0)
        self.assertIn("no drift", format_drift(report))

    def test_directly_imported_array_field_matches_migration(self) -> None:
        blog = self.root / "blog"
        (blog / "models.py").write_text(
            "from django.contrib.postgres.fields import ArrayField\n"
            "from django.db import models\n\n"
            "class Post(models.Model):\n"
            "    title = models.CharField(max_length=100)\n"
            "    tags = ArrayField(models.TextField(), default=list)\n",
            encoding="utf-8",
        )
        (blog / "migrations" / "0002_add_tags.py").write_text(
            "from django.contrib.postgres.fields import ArrayField\n"
            + _migration(
                '        migrations.AddField(model_name="post", name="tags", '
                "field=ArrayField(models.TextField(), default=list)),\n",
                '[("blog", "0001_initial")]',
            ),
            encoding="utf-8",
        )

        report = detect_drift(self.root)
        self.assertEqual(report.drifted, [])

    def test_declared_but_unmigrated_field_blocks(self) -> None:
        report = self._detect({"blog": {"post": {"title", "slug"}}})
        self.assertEqual(len(report.drifted), 1)
        d = report.drifted[0]
        self.assertEqual(d.missing_in_migrations, ["slug"])
        self.assertTrue(d.is_blocking)
        self.assertIn("declared but never migrated", format_drift(report))

    def test_migrated_but_undeclared_field_does_not_block(self) -> None:
        report = self._detect({"blog": {"post": set()}})
        d = report.drifted[0]
        self.assertEqual(d.missing_in_models, ["title"])
        self.assertFalse(d.is_blocking)
        self.assertEqual(report.blocking_count, 0)

    def test_model_with_no_migration_blocks(self) -> None:
        report = self._detect({"blog": {"post": {"title"}, "tag": {"name"}}})
        tag = next(d for d in report.drifted if d.model == "tag")
        self.assertTrue(tag.only_in_models)
        self.assertTrue(tag.is_blocking)

    def test_model_only_in_migrations_does_not_block(self) -> None:
        report = self._detect({"blog": {}})
        d = report.drifted[0]
        self.assertTrue(d.only_in_migrations)
        self.assertFalse(d.is_blocking)

    # --- legend (#57) ----------------------------------------------------
    # Reported by @sevdog: `!!` and `~` were only discoverable by reading the
    # source. Documenting them is worthless if the text and the marks can
    # drift apart later, so these pin the legend to what is actually printed.

    def test_legend_explains_the_mark_it_prints(self) -> None:
        blocking = self._detect({"blog": {"post": {"title", "slug"}}})
        text = format_drift(blocking)
        legend = text.split("\n")[0]
        self.assertTrue(legend.startswith(BLOCKING_MARK), legend)
        self.assertIn("blocking", legend)
        # the mark the legend describes is the mark the entry carries
        entry = next(
            line for line in text.split("\n") if "blog.post" in line
        )
        self.assertTrue(entry.startswith(BLOCKING_MARK), entry)

    def test_advisory_legend_matches_advisory_entry(self) -> None:
        advisory = self._detect({"blog": {"post": set()}})
        text = format_drift(advisory)
        entry = next(line for line in text.split("\n") if "blog.post" in line)
        self.assertTrue(entry.startswith(ADVISORY_MARK), entry)
        self.assertIn(f"{ADVISORY_MARK} advisory", text)

    def test_no_drift_output_carries_no_legend(self) -> None:
        # Nothing is marked, so a legend would be noise on the happy path.
        text = format_drift(self._detect({"blog": {"post": {"id", "title"}}}))
        for line in LEGEND_LINES:
            if line:
                self.assertNotIn(line, text)

    def test_legend_precedes_the_entries(self) -> None:
        text = format_drift(self._detect({"blog": {"post": {"title", "slug"}}}))
        self.assertLess(
            text.index("blocking —"), text.index("blog.post"), text
        )

    def test_implicit_id_is_never_drift(self) -> None:
        # Declared without `id`, migrated with it — Django adds it either way.
        self.assertEqual(self._detect({"blog": {"post": {"title"}}}).drifted, [])

    def test_blocking_entries_sort_first(self) -> None:
        report = self._detect({"blog": {"post": set(), "tag": {"name"}}})
        self.assertTrue(report.drifted[0].is_blocking)

    def test_json_shape(self) -> None:
        payload = self._detect({"blog": {"post": {"title", "slug"}}}).to_dict()
        self.assertEqual(payload["summary"]["blocking"], 1)
        self.assertEqual(payload["drifted"][0]["missingInMigrations"], ["slug"])
        self.assertTrue(payload["drifted"][0]["blocking"])


class SameAppNameTwiceTest(unittest.TestCase):
    """Two directories with the same app name must not fight each other.

    django-guardian ships ``example_project/core`` and
    ``example_project_custom_group/core``. The parser labels an app by its
    directory, so the *declared* side merged both; the migration side replayed
    each directory separately. One project's migrations were therefore compared
    against both projects' models, and on the real checkout that produced a
    duplicated row and a **false blocking** finding — the worst outcome for a
    tool people put in a CI gate.
    """

    def _project(self, root: Path, holder: str, model: str, field: str) -> None:
        app = root / holder / "core"
        (app / "migrations").mkdir(parents=True, exist_ok=True)
        (app / "migrations" / "__init__.py").write_text("", encoding="utf-8")
        (app / "migrations" / "0001_initial.py").write_text(
            "from django.db import migrations, models\n"
            "\n"
            "class Migration(migrations.Migration):\n"
            "    operations = [\n"
            "        migrations.CreateModel(\n"
            f"            name='{model}',\n"
            "            fields=[\n"
            "                ('id', models.AutoField(primary_key=True)),\n"
            f"                ('{field}', models.CharField(max_length=10)),\n"
            "            ],\n"
            "        ),\n"
            "    ]\n",
            encoding="utf-8",
        )
        (app / "models.py").write_text(
            "from django.db import models\n"
            "\n"
            f"class {model}(models.Model):\n"
            f"    {field} = models.CharField(max_length=10)\n",
            encoding="utf-8",
        )

    def test_no_duplicate_rows_and_no_false_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._project(root, "example_project", "Alpha", "one")
            self._project(root, "example_project_custom_group", "Beta", "two")

            payload = detect_drift(str(root)).to_dict()

            labels = [(d["app"], d["model"]) for d in payload["drifted"]]
            self.assertEqual(
                len(labels), len(set(labels)), f"duplicate rows: {labels}"
            )
            self.assertEqual(
                payload["summary"]["blocking"],
                0,
                "a model migrated in the sibling project must not read as "
                f"unmigrated: {payload['drifted']}",
            )

    def test_a_real_missing_migration_still_blocks(self) -> None:
        """Merging the two sides must not blunt the check."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._project(root, "example_project", "Alpha", "one")
            app = root / "example_project" / "core"
            app.joinpath("models.py").write_text(
                "from django.db import models\n"
                "\n"
                "class Alpha(models.Model):\n"
                "    one = models.CharField(max_length=10)\n"
                "    never_migrated = models.IntegerField(default=0)\n",
                encoding="utf-8",
            )
            payload = detect_drift(str(root)).to_dict()
            self.assertEqual(payload["summary"]["blocking"], 1)
            self.assertEqual(
                payload["drifted"][0]["missingInMigrations"], ["never_migrated"]
            )


class BlockingPolicyTest(unittest.TestCase):
    """The narrow blocking rule is a deliberate contract, so pin it."""

    def test_only_the_dangerous_direction_blocks(self) -> None:
        self.assertTrue(ModelDrift("a", "m", missing_in_migrations=["x"]).is_blocking)
        self.assertTrue(ModelDrift("a", "m", only_in_models=True).is_blocking)
        self.assertFalse(ModelDrift("a", "m", missing_in_models=["x"]).is_blocking)
        self.assertFalse(ModelDrift("a", "m", only_in_migrations=True).is_blocking)


if __name__ == "__main__":
    unittest.main()
