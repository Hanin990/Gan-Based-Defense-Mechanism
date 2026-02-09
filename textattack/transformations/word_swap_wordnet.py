"""
Word Swap by swaping synonyms in WordNet
==========================================================
"""

import nltk
nltk.download('omw-1.4')
# nltk.download('arabic')
from nltk.corpus import wordnet
import random
import textattack
from textattack.transformations.word_swap import WordSwap
import json

# class WordSwapWordNet(WordSwap):
#     """Transforms an input by replacing its words with synonyms provided by
#     replacemet json."""
#     def __init__(self, json_file="/notebooks/Textual-Manifold-Based-Attack/TextDefender/formatted_arabic_word_relationships.json"):
#         # Load the synonyms from the given JSON file.
#         with open(json_file, encoding='utf-8') as f:
#             self.synonym_dict = json.load(f)

#     def _get_replacement_words(self, word, random=False):
#         """
#         Returns a list of synonyms for the given word from the JSON mapping.
#         If the word is not found, returns an empty list.
#         """
#         # Get synonyms from the dictionary; if not found, return an empty list.
#         synonyms = self.synonym_dict.get(word, [])
#         # Optionally filter out any instance of the word itself.
#         return [syn for syn in synonyms if syn != word]

class WordSwapWordNet(WordSwap):
    """Transforms an input by replacing its words with synonyms provided by
    WordNet."""

    def __init__(self, language="arb"):
        if language not in wordnet.langs():
            raise ValueError(f"Language {language} not one of {wordnet.langs()}")
        self.language = language
        self.word_list = self._create_wordnet_word_list()
        
    
    
    def _create_wordnet_word_list(self):
        """Create a list of unique words from WordNet."""
        word_set = set()
        for synset in wordnet.all_synsets(lang=self.language):
            for lemma in synset.lemmas(lang=self.language):
                word_set.add(lemma.name().replace('_', ' '))
        return list(word_set)

    # def _get_replacement_words(self, word,  use_random=False):
    #     """Returns a list containing a random word from WordNet."""
    #     random_word = random.choice(self.word_list)
    #     while random_word == word:
    #         random_word = random.choice(self.word_list)
    #     return [random_word]

    def _get_replacement_words(self, word, random=False):
        """Returns a list containing all possible words with 1 character
        replaced by a homoglyph."""
        synonyms = set()
        for syn in wordnet.synsets(word, lang=self.language):
            for syn_word in syn.lemma_names(lang=self.language):
                if (
                    (syn_word != word)
                    and ("_" not in syn_word)
                    and (textattack.shared.utils.is_one_word(syn_word))
                ):
                    # WordNet can suggest phrases that are joined by '_' but we ignore phrases.
                    synonyms.add(syn_word)
        return list(synonyms)
