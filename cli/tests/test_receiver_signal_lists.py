"""Regression tests for positional and keyword ``@receiver`` signals.

Django's receiver accepts one signal or a list/tuple of them. The parser used
to read the first positional argument as a single dotted name, so a list gave
one ``<unknown>`` edge. That also hid list-form ``post_delete`` receivers from
the DB_CASCADE warning.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from orm_lens.signals_parser import signal_graph, skipped_delete_receivers


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


SIGNALS = """from django.db.models.signals import post_delete, post_save
from django.dispatch import receiver

from .models import Attachment


@receiver([post_save, post_delete], sender=Attachment)
def refresh_cache(sender, instance, **kwargs):
    pass


@receiver((post_delete,), sender=Attachment)
def remove_file(sender, instance, **kwargs):
    instance.file.delete(save=False)


@receiver([], sender=Attachment)
def nothing(sender, instance, **kwargs):
    pass
"""


class ReceiverSignalListTest(unittest.TestCase):
    def _graph(self, root: Path) -> dict:
        _write(root / "media" / "signals.py", SIGNALS)
        return signal_graph(str(root))

    def test_list_and_tuple_produce_one_edge_per_signal(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root / "media" / "models.py",
                "from django.db import models\n\n"
                "class Attachment(models.Model):\n"
                "    file = models.FileField()\n",
            )
            result = self._graph(root)
            edges = [(e["signal"], e["handler"].rsplit(".", 1)[-1]) for e in result["edges"]]
            self.assertEqual(
                edges,
                [
                    ("post_save", "refresh_cache"),
                    ("post_delete", "refresh_cache"),
                    ("post_delete", "remove_file"),
                    ("<unknown>", "nothing"),
                ],
            )
            self.assertTrue(all(e.get("builtin") for e in result["edges"][:3]))
            self.assertEqual(result["edges"][0]["sender"], "media.Attachment")

    def test_unreadable_elements_are_skipped(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root / "media" / "models.py",
                "from django.db import models\n\n"
                "class Attachment(models.Model):\n"
                "    file = models.FileField()\n",
            )
            _write(
                root / "media" / "signals.py",
                "from django.db.models.signals import post_delete, post_save\n"
                "from django.dispatch import receiver\n"
                "from .models import Attachment\n\n"
                "@receiver([post_save, make_signal(), post_delete], sender=Attachment)\n"
                "def refresh_cache(sender, instance, **kwargs):\n"
                "    pass\n",
            )
            result = signal_graph(str(root))
            self.assertEqual(
                [e["signal"] for e in result["edges"]],
                ["post_save", "post_delete"],
            )

    def test_list_form_post_delete_appears_in_skipped_receivers(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root / "media" / "models.py",
                "from django.db import models\n\n"
                "class Album(models.Model):\n"
                "    title = models.CharField(max_length=100)\n\n"
                "class Attachment(models.Model):\n"
                "    album = models.ForeignKey(Album, on_delete=models.DB_CASCADE)\n"
                "    file = models.FileField()\n",
            )
            self._graph(root)
            cascade = {
                "cascade_kills": [
                    {"model": "media.Attachment", "db_level": True},
                ]
            }
            result = skipped_delete_receivers(cascade, str(root))
            hits = result["cascade_kills"][0]["skipped_receivers"]
            self.assertEqual(
                {(h["signal"], h["handler"].rsplit(".", 1)[-1]) for h in hits},
                {
                    ("post_delete", "refresh_cache"),
                    ("post_delete", "remove_file"),
                },
            )


class ReceiverSignalKeywordTest(unittest.TestCase):
    def _edges(self, argument: str) -> list[dict]:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root / "media" / "models.py",
                "from django.db import models\n\n"
                "class Attachment(models.Model):\n"
                "    file = models.FileField()\n",
            )
            _write(
                root / "media" / "signals.py",
                "from django.db.models.signals import post_delete, post_save\n"
                "from django.dispatch import Signal, receiver\n"
                "from .models import Attachment\n\n"
                "periodic_task = Signal()\n\n"
                f"@receiver({argument}, sender=Attachment)\n"
                "def handle_change(sender, instance, **kwargs):\n"
                "    pass\n",
            )
            return signal_graph(str(root))["edges"]

    def test_keyword_signal_names_and_builtin_flags(self) -> None:
        cases = [
            ("signal=post_save", [("post_save", True)]),
            ("signal=periodic_task", [("periodic_task", False)]),
            (
                "signal=[post_save, periodic_task, post_delete]",
                [("post_save", True), ("periodic_task", False), ("post_delete", True)],
            ),
            ("signal=(post_delete,)", [("post_delete", True)]),
        ]
        for argument, expected in cases:
            with self.subTest(argument=argument):
                edges = self._edges(argument)
                self.assertEqual(
                    [(e["signal"], e.get("builtin", False)) for e in edges], expected
                )
                self.assertTrue(all(e["sender"] == "media.Attachment" for e in edges))

    def test_keyword_unreadable_elements_and_missing_signal(self) -> None:
        cases = [
            ("signal=[]", ["<unknown>"]),
            ("signal=()", ["<unknown>"]),
            ("signal=make_signal()", ["<unknown>"]),
            ("signal=[post_save, make_signal(), post_delete]", ["post_save", "post_delete"]),
            ("dispatch_uid='refresh'", ["<unknown>"]),
        ]
        for argument, expected in cases:
            with self.subTest(argument=argument):
                self.assertEqual([e["signal"] for e in self._edges(argument)], expected)
