import spacy
from spacy import displacy
import en_core_web_sm

def dependency():
    nlp = en_core_web_sm.load()
    doc = nlp(u'This is a sentence.')
    return displacy.serve(doc, style='dep')

