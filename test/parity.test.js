const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const test = require('node:test');

const originalLoad = Module._load;
Module._load = function (request, parent, isMain) {
  if (request === 'vscode') return {};
  return originalLoad.call(this, request, parent, isMain);
};

const { parseModelsFile } = require('../out/parser');

test.after(() => {
  Module._load = originalLoad;
});

const fixturePath = path.join(__dirname, 'fixtures', 'parity_input.py');
const fixtureSource = fs.readFileSync(fixturePath, 'utf-8');

test('parity fixture — model shape matches golden expectations', () => {
  const models = parseModelsFile(fixturePath, fixtureSource);

  assert.equal(models.length, 5, 'five concrete model classes');
  assert.deepEqual(
    models.map((m) => m.name),
    ['Author', 'Book', 'Tag', 'BookTag', 'Chapter'],
    'Auditable is abstract and has no table, so it is not among them'
  );

  // The shape the fixture was missing until 0.17.0, and the reason both
  // suites were green while the two parsers disagreed: `Chapter(Auditable)`
  // is only a model because its base is, and its `created` column only exists
  // because that base is abstract.
  const chapter = models[4];
  assert.deepEqual(
    chapter.fields.map((f) => f.name),
    ['heading', 'book', 'updated'],
    'fields holds only what Chapter declares'
  );
  assert.deepEqual(
    chapter.inheritedFields.map((f) => f.name),
    ['created'],
    'updated is redeclared on Chapter, so it shadows the base'
  );
  assert.equal(chapter.inheritedFields[0].inheritedFrom, 'Auditable');

  for (const m of models) {
    assert.equal(m.appName, 'fixtures', `${m.name}: appName derived from parent dir`);
  }

  const author = models[0];
  assert.deepEqual(author.fields.map(({ name, type }) => [name, type]), [
    ['name', 'CharField'],
    ['aliases', 'ArrayField'],
    ['tags', 'ArrayField'],
    ['punctuation', 'CharField'],
    ['blurb', 'TextField'],
    ['attributes', 'HStoreField'],
    ['years_active', 'IntegerRangeField'],
    ['search_document', 'SearchVectorField'],
    ['normalized_name', 'GeneratedField'],
  ]);
  assert.ok(author.fields.every((field) => field.isRelation === false));
  assert.equal(author.fields[1].args, 'ArrayField(models.TextField(), size=2)');

  assert.equal(author.fields[2].args, 'models.CharField(max_length=10, default="(")');
  assert.equal(author.fields[3].args, String.raw`max_length=10, default='it\'s )'`);
  assert.equal(author.fields[4].args, 'default="""first (\nsecond (\n"""');

  const book = models[1];
  assert.equal(book.fields.length, 4, 'title + author + editor + tags');
  assert.deepEqual(book.meta, { ordering: "['title']" });

  const authorField = book.fields.find((f) => f.name === 'author');
  assert.equal(authorField.type, 'ForeignKey');
  assert.equal(authorField.isRelation, true);
  assert.equal(authorField.relationKind, 'ForeignKey');
  assert.equal(authorField.relatedModel, 'Author');
  assert.equal(authorField.onDelete, 'CASCADE');
  assert.equal(authorField.relatedName, 'books');

  const editorField = book.fields.find((f) => f.name === 'editor');
  assert.equal(editorField.relationKind, 'OneToOneField');
  assert.equal(editorField.onDelete, 'SET_NULL');
  assert.equal(editorField.relatedName, 'edited_book');

  const tagsField = book.fields.find((f) => f.name === 'tags');
  assert.equal(tagsField.relationKind, 'ManyToManyField');
  assert.equal(tagsField.relatedModel, 'Tag');
  assert.equal(tagsField.throughModel, 'BookTag');

  const bookTag = models[3];
  assert.equal(bookTag.fields.length, 2);
  assert.equal(bookTag.fields[0].name, 'book');
  assert.equal(bookTag.fields[0].onDelete, 'CASCADE');
  assert.equal(bookTag.fields[1].name, 'tag');
});
