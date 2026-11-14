from django.contrib.postgres.fields import ArrayField, HStoreField, IntegerRangeField
from django.contrib.postgres.search import SearchVectorField
from django.db import models
from django.db.models import GeneratedField
from django.db.models.functions import Lower

class Author(models.Model):
    name = models.CharField(max_length=100)
    aliases = ArrayField(ArrayField(models.TextField(), size=2))
    tags = ArrayField(models.CharField(max_length=10, default="("))
    punctuation = models.CharField(max_length=10, default='it\'s )')
    blurb = models.TextField(default="""first (
second (
""")
    attributes = HStoreField(default=dict)
    years_active = IntegerRangeField(null=True)
    search_document = SearchVectorField(null=True)
    normalized_name = GeneratedField(
        expression=Lower("name"), output_field=models.TextField(), db_persist=True
    )


class Book(models.Model):
    title = models.CharField(max_length=200)
    author = models.ForeignKey('Author', on_delete=models.CASCADE, related_name='books')
    editor = models.OneToOneField('Author', on_delete=models.SET_NULL, related_name='edited_book')
    tags = models.ManyToManyField('Tag', through='BookTag')

    class Meta:
        ordering = ['title']


class Tag(models.Model):
    name = models.SlugField()


class BookTag(models.Model):
    book = models.ForeignKey('Book', on_delete=models.CASCADE)
    tag = models.ForeignKey('Tag', on_delete=models.CASCADE)


# A project-local abstract base and a concrete subclass of it. Present because
# both parsers were green on this fixture while disagreeing about exactly this
# shape: the TypeScript side did not recognise `Chapter` as a model at all,
# and neither suite could see that, because the fixture had no local base in
# it. `class X(SomeLocalBase)` is about as common as Django patterns get.
class Auditable(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class Chapter(Auditable):
    heading = models.CharField(max_length=120)
    book = models.ForeignKey('Book', on_delete=models.CASCADE, related_name='chapters')
    updated = models.CharField(max_length=10)
