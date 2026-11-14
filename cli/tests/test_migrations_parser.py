"""Tests for the static migration DAG parser (describe_migration_dependency).

The parser reads ``<app>/migrations/*.py`` via pure ``ast`` — no Django boot,
no DB. These tests build synthetic migration folders inside ``tmp_path`` and
assert the returned DAG shape: dependencies, roots, leaves, cross-app deps,
and the structured missing-folder response.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from orm_lens.migrations_parser import (
    _migration_prefix,
    describe_migration_dependency,
)


def _write_migration(app_dir: Path, name: str, body: str) -> None:
    (app_dir / "migrations").mkdir(parents=True, exist_ok=True)
    (app_dir / "migrations" / f"{name}.py").write_text(body, encoding="utf-8")


def _init_py(app_dir: Path) -> None:
    """Emulate a real migrations package layout."""
    (app_dir / "migrations" / "__init__.py").write_text("", encoding="utf-8")


class DescribeMigrationDependencyTest(unittest.TestCase):
    def test_simple_chain_roots_and_leaves(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_dir = root / "orders"
            _write_migration(
                app_dir,
                "0001_initial",
                (
                    "from django.db import migrations, models\n\n"
                    "class Migration(migrations.Migration):\n"
                    "    initial = True\n"
                    "    dependencies = []\n"
                    "    operations = [\n"
                    "        migrations.CreateModel(name='Order', fields=[]),\n"
                    "        migrations.CreateModel(name='LineItem', fields=[]),\n"
                    "    ]\n"
                ),
            )
            _write_migration(
                app_dir,
                "0002_add_status",
                (
                    "from django.db import migrations\n\n"
                    "class Migration(migrations.Migration):\n"
                    "    dependencies = [('orders', '0001_initial')]\n"
                    "    operations = [migrations.AddField(model_name='order', name='status')]\n"
                ),
            )
            _init_py(app_dir)

            result = describe_migration_dependency("orders", str(root))

            self.assertEqual(result["app_label"], "orders")
            self.assertNotIn("error", result)
            self.assertEqual(
                [m["name"] for m in result["migrations"]],
                ["0001_initial", "0002_add_status"],
            )
            self.assertEqual(result["roots"], ["0001_initial"])
            self.assertEqual(result["leaves"], ["0002_add_status"])
            self.assertEqual(result["cross_app_deps"], [])

            initial = result["migrations"][0]
            self.assertEqual(initial["dependencies"], [])
            self.assertEqual(initial["operations"], ["CreateModel", "CreateModel"])

            second = result["migrations"][1]
            self.assertEqual(second["dependencies"], [["orders", "0001_initial"]])
            self.assertEqual(second["operations"], ["AddField"])

    def test_cross_app_dep_flagged(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_dir = root / "orders"
            _write_migration(
                app_dir,
                "0001_initial",
                (
                    "from django.db import migrations\n\n"
                    "class Migration(migrations.Migration):\n"
                    "    dependencies = []\n"
                    "    operations = []\n"
                ),
            )
            _write_migration(
                app_dir,
                "0002_link_user",
                (
                    "from django.db import migrations\n\n"
                    "class Migration(migrations.Migration):\n"
                    "    dependencies = [\n"
                    "        ('orders', '0001_initial'),\n"
                    "        ('users', '0003_alter_email'),\n"
                    "    ]\n"
                    "    operations = [migrations.AddField(model_name='order', name='user')]\n"
                ),
            )
            _init_py(app_dir)

            result = describe_migration_dependency("orders", str(root))

            self.assertEqual(result["cross_app_deps"], [["users", "0003_alter_email"]])
            self.assertEqual(result["roots"], ["0001_initial"])
            self.assertEqual(result["leaves"], ["0002_link_user"])

    def test_missing_folder_returns_structured_error(self) -> None:
        with TemporaryDirectory() as tmp:
            result = describe_migration_dependency("nonexistent_app", tmp)
            self.assertEqual(result["app_label"], "nonexistent_app")
            self.assertEqual(result["migrations"], [])
            self.assertEqual(result["error"], "no migrations folder found")

    def test_bare_operation_import_and_multi_leaf(self) -> None:
        """Bare-import ``AddField(...)`` form + two independent tips → two leaves."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_dir = root / "billing"
            _write_migration(
                app_dir,
                "0001_initial",
                (
                    "from django.db import migrations\n"
                    "from django.db.migrations import AddField\n\n"
                    "class Migration(migrations.Migration):\n"
                    "    dependencies = []\n"
                    "    operations = [AddField(model_name='invoice', name='total')]\n"
                ),
            )
            _write_migration(
                app_dir,
                "0002_branch_a",
                (
                    "from django.db import migrations\n\n"
                    "class Migration(migrations.Migration):\n"
                    "    dependencies = [('billing', '0001_initial')]\n"
                    "    operations = []\n"
                ),
            )
            _write_migration(
                app_dir,
                "0003_branch_b",
                (
                    "from django.db import migrations\n\n"
                    "class Migration(migrations.Migration):\n"
                    "    dependencies = [('billing', '0001_initial')]\n"
                    "    operations = []\n"
                ),
            )
            _init_py(app_dir)

            result = describe_migration_dependency("billing", str(root))

            self.assertEqual(result["roots"], ["0001_initial"])
            self.assertEqual(
                sorted(result["leaves"]),
                ["0002_branch_a", "0003_branch_b"],
            )
            self.assertEqual(result["migrations"][0]["operations"], ["AddField"])


class NonStandardFilenameTest(unittest.TestCase):
    """Django imports every module in ``migrations/`` that does not start with
    ``_`` or ``~`` — a four-digit prefix is not required."""

    def test_hand_edited_number_keeps_its_numeric_position(self) -> None:
        # A human appended a letter to keep `0162b` next to `0162`; the file is
        # still #162, not an unorderable name that sorts before `0001`.
        self.assertEqual(_migration_prefix("0162b_auto_20201218_1810"), 162)
        self.assertEqual(_migration_prefix("008_remove_scheduled_mails"), 8)
        self.assertEqual(_migration_prefix("0002_add_status"), 2)
        self.assertEqual(_migration_prefix("initial"), -1)

    def test_hand_edited_and_short_files_join_the_graph(self) -> None:
        with TemporaryDirectory() as tmp:
            app_dir = Path(tmp) / "shop"
            _write_migration(
                app_dir,
                "0001_initial",
                "from django.db import migrations\n\n"
                "class Migration(migrations.Migration):\n"
                "    dependencies = []\n"
                "    operations = []\n",
            )
            _write_migration(
                app_dir,
                "0001b_add_session_key",
                "from django.db import migrations\n\n"
                "class Migration(migrations.Migration):\n"
                "    dependencies = [('shop', '0001_initial')]\n"
                "    operations = [migrations.AddField(model_name='post',"
                " name='session_key', field=None)]\n",
            )
            _write_migration(
                app_dir,
                "008_legacy",
                "from django.db import migrations\n\n"
                "class Migration(migrations.Migration):\n"
                "    dependencies = [('shop', '0001b_add_session_key')]\n"
                "    operations = []\n",
            )
            _init_py(app_dir)

            result = describe_migration_dependency("shop", str(tmp))

            names = [m["name"] for m in result["migrations"]]
            self.assertIn("0001b_add_session_key", names)
            self.assertIn("008_legacy", names)
            # Without the files the graph had three roots and two extra leaves.
            self.assertEqual(result["roots"], ["0001_initial"])
            self.assertEqual(result["leaves"], ["008_legacy"])

    def test_helper_module_is_not_a_migration_node(self) -> None:
        """Django imports any module in the package; only `Migration` is one."""
        with TemporaryDirectory() as tmp:
            app_dir = Path(tmp) / "shop"
            _write_migration(
                app_dir,
                "0001_initial",
                "from django.db import migrations\n\n"
                "class Migration(migrations.Migration):\n"
                "    dependencies = []\n"
                "    operations = []\n",
            )
            _write_migration(app_dir, "helpers", "NEW_FIELDS = []\n")
            _init_py(app_dir)

            result = describe_migration_dependency("shop", str(tmp))

            self.assertEqual([m["name"] for m in result["migrations"]], ["0001_initial"])
            self.assertEqual(result["leaves"], ["0001_initial"])

    def test_reassigned_dependencies_use_the_final_value(self) -> None:
        """Python keeps the last class attribute; Django orders by that one."""
        with TemporaryDirectory() as tmp:
            app_dir = Path(tmp) / "shop"
            _write_migration(
                app_dir,
                "0001_initial",
                "from django.db import migrations\n\n"
                "class Migration(migrations.Migration):\n"
                "    dependencies = []\n"
                "    operations = []\n",
            )
            _write_migration(
                app_dir,
                "0002_add_slug",
                "from django.db import migrations\n\n"
                "class Migration(migrations.Migration):\n"
                "    dependencies = []\n"
                "    dependencies = [('shop', '0001_initial')]\n"
                "    operations = []\n",
            )
            _init_py(app_dir)

            result = describe_migration_dependency("shop", str(tmp))

            by_name = {m["name"]: m for m in result["migrations"]}
            self.assertEqual(by_name["0002_add_slug"]["dependencies"], [["shop", "0001_initial"]])
            self.assertEqual(result["roots"], ["0001_initial"])


if __name__ == "__main__":
    unittest.main()
