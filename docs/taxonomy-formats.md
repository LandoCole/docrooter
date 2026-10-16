# Taxonomy formats

A taxonomy is a tree of categories. `Taxonomy.load(path)` and the CLI read it from JSON or CSV
and detect the layout themselves; `Taxonomy.from_dict(data)` takes the same JSON structures
as Python objects. There are five layouts, and all five below describe the same taxonomy (two
top-level categories, two children each). Loading any of them gives a taxonomy with the same
`content_hash()`.

Check a file before using it:

```bash
docrooter taxonomy validate taxonomy.csv
# valid: 6 nodes, depth 2
```

The command prints every problem it finds (with the row or JSON pointer it came from) and exits
with status 1 if any is an error.

## JSON

A JSON file is an object with a `nodes` list (or just the list). Each node has a `label`; `id`
is optional.

### Nested

Children sit inside their parent, so `parent_id` is implied.

```json taxonomy
{"nodes": [
  {"id": "financial", "label": "Financial", "description": "Money, accounting, results",
   "children": [
     {"id": "financial.tax", "label": "Tax", "description": "Income tax and deferred tax"},
     {"id": "financial.audit", "label": "Audit", "description": "Audit reports and findings"}
   ]},
  {"id": "legal", "label": "Legal", "description": "Legal matters",
   "children": [
     {"id": "legal.contracts", "label": "Contracts", "description": "Agreements and obligations"},
     {"id": "legal.litigation", "label": "Litigation", "description": "Disputes and proceedings"}
   ]}
]}
```

### Flat

One list of nodes, linked by `parent_id`. A node without a `parent_id` is a top-level category.

```json taxonomy
{"nodes": [
  {"id": "financial", "label": "Financial", "parent_id": null, "description": "Money, accounting, results"},
  {"id": "financial.tax", "label": "Tax", "parent_id": "financial", "description": "Income tax and deferred tax"},
  {"id": "financial.audit", "label": "Audit", "parent_id": "financial", "description": "Audit reports and findings"},
  {"id": "legal", "label": "Legal", "description": "Legal matters"},
  {"id": "legal.contracts", "label": "Contracts", "parent_id": "legal", "description": "Agreements and obligations"},
  {"id": "legal.litigation", "label": "Litigation", "parent_id": "legal", "description": "Disputes and proceedings"}
]}
```

Nested and flat can be mixed. If a nested node also gives a `parent_id`, it has to match the
node it sits in.

### Node fields

| Field | Meaning |
| --- | --- |
| `label` | Required. The category's name, shown in results and matched case-insensitively in labelled datasets. |
| `id` | Stable identifier. Defaults to the slugged label path, e.g. `financial.tax`. |
| `description` | Text that tells a backend what belongs here. Zero-shot backends rely on it. |
| `examples` | A list of example texts. Backends that take examples use them. |
| `parent_id` | The parent's id; `null` or missing for a top-level category. |
| `children` | Nested child nodes. |
| `multi_label` | If true, the decision among this node's children may pick several. Default false. |
| `threshold` | Minimum probability to follow a child of this node. Overrides the classifier's `threshold`. |
| `backend` | Backend spec (e.g. `"jev"`) for decisions among this node's children. |
| `metadata` | Any JSON object. Carried along; docrooter does not use it. |

### Root settings

The decision among the top-level categories has no node of its own. Give it settings with a
top-level `root` object (object form only). It accepts `multi_label`, `threshold`, `backend` and
`description`.

```json taxonomy-any
{"root": {"multi_label": true, "threshold": 0.3, "backend": "jev"},
 "nodes": [
  {"id": "a", "label": "A"},
  {"id": "b", "label": "B", "children": [{"id": "b.1", "label": "One"}, {"id": "b.2", "label": "Two"}]}
]}
```

## CSV

CSV files are read as UTF-8 (a byte-order mark is fine). The layout is detected from the header:
a `parent_id` column means adjacency, a `level_1` column means levels, a `path` column means
paths. Header names are case-insensitive. Blank rows are skipped and cells are trimmed.

All three layouts accept these optional columns: `description`, `examples` (several examples
separated by `|`), `backend`, `threshold` and `multi_label` (`true`/`false`, `yes`/`no`, `1`/`0`).
Other columns are ignored, so an export with stray empty columns still loads.

### Adjacency

One row per node, linked by `parent_id`. Needs `label`; `id` is optional (it defaults to the
parent's id plus the slugged label).

```csv taxonomy
id,parent_id,label,description
financial,,Financial,"Money, accounting, results"
financial.tax,financial,Tax,Income tax and deferred tax
financial.audit,financial,Audit,Audit reports and findings
legal,,Legal,Legal matters
legal.contracts,legal,Contracts,Agreements and obligations
legal.litigation,legal,Litigation,Disputes and proceedings
```

### Levels

One column per level (`level_1`, `level_2`, ...), the way a spreadsheet usually holds a tree.
A row names the category at its deepest filled level, and the ancestors it implies are created
if they have no row of their own. Ids are the slugged labels joined by dots.

```csv taxonomy
level_1,level_2,description
Financial,,"Money, accounting, results"
Financial,Tax,Income tax and deferred tax
Financial,Audit,Audit reports and findings
Legal,,Legal matters
Legal,Contracts,Agreements and obligations
Legal,Litigation,Disputes and proceedings
```

### Path

One row per node with the labels joined by `>`. Use `--sep` on the CLI (or `sep=` in
`Taxonomy.load`) if your labels contain `>`. Ancestors and ids work as in the levels layout.

```csv taxonomy
path,description
Financial,"Money, accounting, results"
Financial > Tax,Income tax and deferred tax
Financial > Audit,Audit reports and findings
Legal,Legal matters
Legal > Contracts,Agreements and obligations
Legal > Litigation,Disputes and proceedings
```

## Spreadsheet exports

Excel and Google Sheets exports usually carry a byte-order mark, padded cells, blank trailing
rows and empty trailing columns. All of that is tolerated: cells are trimmed, blank rows are
dropped and columns that are not part of the layout are ignored.

## What validation checks

Errors stop a file from loading; warnings are reported but allowed.

| Code | Severity | Meaning |
| --- | --- | --- |
| `empty_taxonomy` | error | There are no nodes. |
| `duplicate_id` | error | Two nodes share an id. |
| `reserved_id` | error | A node uses `__root__`, which names the virtual root. |
| `empty_label` | error | A node has a blank label. |
| `missing_parent` | error | A `parent_id` names a node that does not exist. |
| `cycle` | error | Parent links form a loop. |
| `duplicate_sibling_label` | error | Two siblings have the same label once case and spacing are normalised. |
| `single_child` | warning | A node has exactly one child, which adds a step with no decision. |
| `invalid_json`, `invalid_structure`, `invalid_node`, `parent_mismatch` | error | The JSON could not be read as a taxonomy. |
| `unknown_csv_format`, `empty_path`, `empty_path_segment`, `invalid_value` | error | The CSV header or a row could not be read. |
