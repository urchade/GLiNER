import spacy
from spacy.tokens import Doc
from spacy import displacy
from spacy.cli import download
import sys
import os
from model import GLiNER


class GLiNERPipeline:
    def __init__(self, labels, chunk_size=250, style="ent", spacy_model="en_core_web_sm"):
        # Store the model and configuration
        self.model = GLiNER.from_pretrained("urchade/gliner_base")
        self.labels = labels
        self.chunk_size = chunk_size
        self.style = style  # Store style as an instance variable

        # Check if the spaCy model is available, download if not
        try:
            self.nlp = spacy.load(spacy_model, disable=["ner"])
        except OSError:
            print(f"Downloading spaCy model '{spacy_model}'...")
            download(spacy_model)
            self.nlp = spacy.load(spacy_model, disable=["ner"])

    def __call__(self, text):
        # Tokenize the text
        doc = self.nlp(text)
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size if start + self.chunk_size < len(text) else len(text)
            # Ensure the chunk ends at a complete word
            while end < len(text) and text[end] not in [' ', '\n']:
                end += 1
            chunks.append(text[start:end])
            start = end

        # Process each chunk and adjust entity indices
        all_entities = []
        offset = 0
        for chunk in chunks:
            if self.style == "span":
                chunk_entities = self.model.predict_entities(chunk, self.labels, flat_ner=False)
            else:
                chunk_entities = self.model.predict_entities(chunk, self.labels, flat_ner=True)
            for entity in chunk_entities:
                all_entities.append({
                    'start': offset + entity['start'],
                    'end': offset + entity['end'],
                    'label': entity['label']
                })
            offset += len(chunk)

        # Create new spans for the entities and add them to the doc
        doc = self._create_entity_spans(doc, all_entities)

        return doc

    def _create_entity_spans(self, doc, all_entities):
        spans = []
        for ent in all_entities:
            span = doc.char_span(ent['start'], ent['end'], label=ent['label'])
            if span:  # Only add span if it is valid
                spans.append(span)
        if self.style == "span":
            doc.spans["sc"] = spans
        else:
            doc.ents = spans
        return doc

    def visualize(self, doc):
        # Visualize the entities using displacy
        if self.style == "span":
            displacy.render(doc, style="span", options={"spans_key": "sc"})
        else:
            displacy.render(doc, style="ent")