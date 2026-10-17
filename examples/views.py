"""Views: the text a backend actually sees, for a section and for a whole document.

Needs no models. Backends never read raw sections; they read a budgeted view, so printing
the view is the first thing to check when a label looks wrong.
"""

from docrooter import Classifier, DocumentView, SectionView, Taxonomy, load

MARKDOWN = """\
# Annual report

## Income taxes
The effective tax rate was 21.4 percent. Deferred tax assets rose.

## Risk factors
Our business faces risks from competition and regulation.
"""

doc = load(MARKDOWN)

# A section view: which fields, in which order, within how many tokens.
view = SectionView(fields=("path", "title", "lead"), max_tokens=64)
for section_id, rendered in view.render_all(doc, [s.id for s in doc.leaves()]).items():
    print(f"[{section_id}] {rendered.n_tokens} tokens, truncated={rendered.truncated}")
    print(rendered.text)

# A document view: one compact text per document, used for the document type.
doc_view = DocumentView(fields=("title", "stats", "toc", "lead"), max_tokens=128)
print(doc_view.render(doc).text)

# Pass a view to the classifier to change what every section contributes.
taxonomy = Taxonomy.from_dict(
    {
        "nodes": [
            {"id": "tax", "label": "Tax", "description": "Income tax and the tax rate"},
            {"id": "risk", "label": "Risk", "description": "Risk factors and competition"},
        ]
    }
)
classifier = Classifier(taxonomy, "keyword", view=view, threshold=0.0)
for section in classifier.classify(doc).sections.values():
    print(f"{section.title}: {section.best.labels[-1]}")
