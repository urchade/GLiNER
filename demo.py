from typing import Dict, Union
from gliner import GLiNER
import gradio as gr

model = GLiNER.from_pretrained("model/", load_tokenizer=True)

examples = [
    [
        "Libretto by Marius Petipa, based on the 1822 novella ``Trilby, ou Le Lutin d'Argail`` by Charles Nodier, first presented by the Ballet of the Moscow Imperial Bolshoi Theatre on January 25/February 6 (Julian/Gregorian calendar dates), 1870, in Moscow with Polina Karpakova as Trilby and Ludiia Geiten as Miranda and restaged by Petipa for the Imperial Ballet at the Imperial Bolshoi Kamenny Theatre on January 17–29, 1871 in St. Petersburg with Adèle Grantzow as Trilby and Lev Ivanov as Count Leopold.",
        "person, book, location, date, actor, character",
        0.3,
        True,
    ],
    [
        """
* Data Scientist, Data Analyst, or Data Engineer with 1+ years of experience.
* Experience with technologies such as Docker, Kubernetes, or Kubeflow
* Machine Learning experience preferred
* Experience with programming languages such as Python, C++, or SQL preferred
* Experience with technologies such as Databricks, Qlik, TensorFlow, PyTorch, Python, Dash, Pandas, or NumPy preferred
* BA or BS degree
* Active Secret OR Active Top Secret or Active TS/SCI clearance
""",
        "software package, programing language, software tool, degree, job title",
        0.3,
        False,
    ],
    [
        "However, both models lack other frequent DM symptoms including the fibre-type dependent atrophy, myotonia, cataract and male-infertility.",
        "disease, symptom",
        0.3,
        False,
    ],
    [
        "Synergy between signal transduction pathways is obligatory for expression of c-fos in B and T cell lines: implication for c-fos control via surface immunoglobulin and T cell antigen receptors.",
        "DNA, RNA, cell line, cell type, protein",
        0.3,
        False,
    ],
    [
        "The choice of the encoder and decoder modules of dnpg can be quite flexible, for instance long short term memory networks (lstm) or convolutional neural network (cnn).",
        "short acronym, long acronym",
        0.3,
        False,
    ],
    [
        "Amelia Earhart flew her single engine Lockheed Vega 5B across the Atlantic to Paris.",
        "person, company, location, airplane",
        0.3,
        True,
    ],
    [
        "Feldman is a contributor to NBC Sports Boston's ``State of the Revs`` and ``Revolution Postgame Live`` programs as well as to 98.5 the SportsHub, SiriusXM FC's MLS coverage and to other New England and national radio outlets and podcasts.",
        "person, company, location",
        0.3,
        False,
    ],
    [
        "On 25 July 1948, on the 39th anniversary of Bleriot's crossing of the English Channel, the Type 618 Nene-Viking flew Heathrow to Paris (Villacoublay) in the morning carrying letters to Bleriot's widow and son (secretary of the FAI), who met it at the airport.",
        "date, location, person, organization",
        0.3,
        False,
    ],
    [
        "Leo & Ian won the 1962 Bathurst Six Hour Classic at Mount Panorama driving a Daimler SP250 sports car, (that year the 500 mile race for touring cars were held at Phillip Island)",
        "person, date, location, organization, competition",
        0.3,
        False,
    ],
    [
        "The Shore Line route of the CNS & M until 1955 served, from south to north, the Illinois communities of Chicago, Evanston, Wilmette, Kenilworth, Winnetka, Glencoe, Highland Park, Highwood, Fort Sheridan, Lake Forest, Lake Bluff, North Chicago, Waukegan, Zion, and Winthrop Harbor as well as Kenosha, Racine, and Milwaukee (the ``KRM'') in Wisconsin.",
        "location, organization, date",
        0.3,
        False,
    ],
    [
        "Comet C/2006 M4 (SWAN) is a non-periodic comet discovered in late June 2006 by Robert D. Matson of Irvine, California and Michael Mattiazzo of Adelaide, South Australia in publicly available images of the Solar and Heliospheric Observatory (SOHO).",
        "person, organization, date, location",
        0.3,
        False,
    ],
    [
        "From November 29, 2011 to March 31, 2012, Karimloo returned to ``Les Misérables`` to play the lead role of Jean Valjean at The Queen's Theatre, London, for which he won the 2013 Theatregoers' Choice Award for Best Takeover in a Role.",
        "person, actor, award, date, location",
        0.3,
        False,
    ],
    [
        "A Mexicali health clinic supported by former Baja California gubernatorial candidate Enrique Acosta Fregoso (PRI) was closed on June 15 after selling a supposed COVID-19 ``cure'' for between MXN $10,000 and $50,000.",
        "location, organization, person, date, currency",
        0.3,
        False,
    ],
    [
        "Built in 1793, it was the home of Mary Young Pickersgill when she moved to Baltimore in 1806 and the location where she later sewed the ``Star Spangled Banner'', in 1813, the huge out-sized garrison flag that flew over Fort McHenry at Whetstone Point in Baltimore Harbor in the summer of 1814 during the British Royal Navy attack in the Battle of Baltimore during the War of 1812.",
        "date, person, location, organization, event, flag",
        0.3,
        False,
    ],
]


def ner(
    text, labels: str, threshold: float, nested_ner: bool
) -> Dict[str, Union[str, int, float]]:
    labels = labels.split(",")
    return {
        "text": text,
        "entities": [
            {
                "entity": entity["label"],
                "word": entity["text"],
                "start": entity["start"],
                "end": entity["end"],
                "score": 0,
            }
            for entity in model.predict_entities(
                text, labels, flat_ner=not nested_ner, threshold=threshold
            )
        ],
    }


with gr.Blocks(title="GLiNER-M-v2.1") as demo:
    gr.Markdown(
        """
        # GLiNER-base
        GLiNER is a Named Entity Recognition (NER) model capable of identifying any entity type using a bidirectional transformer encoder (BERT-like). It provides a practical alternative to traditional NER models, which are limited to predefined entities, and Large Language Models (LLMs) that, despite their flexibility, are costly and large for resource-constrained scenarios.
        ## Links
        * Model: https://huggingface.co/urchade/gliner_multi-v2.1
        * All GLiNER models: https://huggingface.co/models?library=gliner
        * Paper: https://arxiv.org/abs/2311.08526
        * Repository: https://github.com/urchade/GLiNER
        """
    )
    with gr.Accordion("How to run this model locally", open=False):
        gr.Markdown(
            """
            ## Installation
            To use this model, you must install the GLiNER Python library:
            ```
            !pip install gliner
            ```
         
            ## Usage
            Once you've downloaded the GLiNER library, you can import the GLiNER class. You can then load this model using `GLiNER.from_pretrained` and predict entities with `predict_entities`.
            """
        )
        gr.Code(
            '''
from gliner import GLiNER
model = GLiNER.from_pretrained("urchade/gliner_mediumv2.1")
text = """
Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃˈtjɐnu ʁɔˈnaldu]; born 5 February 1985) is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and the Portugal national team. Widely regarded as one of the greatest players of all time, Ronaldo has won five Ballon d'Or awards,[note 3] a record three UEFA Men's Player of the Year Awards, and four European Golden Shoes, the most by a European player. He has won 33 trophies in his career, including seven league titles, five UEFA Champions Leagues, the UEFA European Championship and the UEFA Nations League. Ronaldo holds the records for most appearances (183), goals (140) and assists (42) in the Champions League, goals in the European Championship (14), international goals (128) and international appearances (205). He is one of the few players to have made over 1,200 professional career appearances, the most by an outfield player, and has scored over 850 official senior career goals for club and country, making him the top goalscorer of all time.
"""
labels = ["person", "award", "date", "competitions", "teams"]
entities = model.predict_entities(text, labels)
for entity in entities:
    print(entity["text"], "=>", entity["label"])
            ''',
            language="python",
        )
        gr.Code(
            """
Cristiano Ronaldo dos Santos Aveiro => person
5 February 1985 => date
Al Nassr => teams
Portugal national team => teams
Ballon d'Or => award
UEFA Men's Player of the Year Awards => award
European Golden Shoes => award
UEFA Champions Leagues => competitions
UEFA European Championship => competitions
UEFA Nations League => competitions
Champions League => competitions
European Championship => competitions
            """
        )

    input_text = gr.Textbox(
        value=examples[0][0], label="Text input", placeholder="Enter your text here"
    )
    with gr.Row() as row:
        labels = gr.Textbox(
            value=examples[0][1],
            label="Labels",
            placeholder="Enter your labels here (comma separated)",
            scale=2,
        )
        threshold = gr.Slider(
            0,
            1,
            value=0.3,
            step=0.01,
            label="Threshold",
            info="Lower the threshold to increase how many entities get predicted.",
            scale=1,
        )
        nested_ner = gr.Checkbox(
            value=examples[0][2],
            label="Nested NER",
            info="Allow for nested NER?",
            scale=0,
        )
    output = gr.HighlightedText(label="Predicted Entities")
    submit_btn = gr.Button("Submit")
    examples = gr.Examples(
        examples,
        fn=ner,
        inputs=[input_text, labels, threshold, nested_ner],
        outputs=output,
        cache_examples=True,
    )

    # Submitting
    input_text.submit(
        fn=ner, inputs=[input_text, labels, threshold, nested_ner], outputs=output
    )
    labels.submit(
        fn=ner, inputs=[input_text, labels, threshold, nested_ner], outputs=output
    )
    threshold.release(
        fn=ner, inputs=[input_text, labels, threshold, nested_ner], outputs=output
    )
    submit_btn.click(
        fn=ner, inputs=[input_text, labels, threshold, nested_ner], outputs=output
    )
    nested_ner.change(
        fn=ner, inputs=[input_text, labels, threshold, nested_ner], outputs=output
    )

demo.queue()
demo.launch(debug=True)
