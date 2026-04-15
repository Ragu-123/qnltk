from dataclasses import dataclass, field
from typing import List, Optional
import logging
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

logger = logging.getLogger(__name__)

DEFAULT_GRAMMAR = r"""
NP: {<DT>?<JJ>*<NN.*>+}
PP: {<IN><NP>}
VP: {<VB.*><NP|PP|CLAUSE>+$}
CLAUSE: {<NP><VP>}
S: {<CLAUSE>+}
"""


@dataclass
class ParseNode:
    """A node in a parse tree."""
    label: str
    children: List['ParseNode'] = field(default_factory=list)
    is_leaf: bool = False
    word: Optional[str] = None
    pos: Optional[str] = None


class OOVHandler:
    """Handles out-of-vocabulary words via character trigram fallback."""

    def __init__(self, embedding_vocab: set):
        self._vocab = embedding_vocab
        self._oov_words: list[str] = []

    @property
    def oov_words(self) -> list[str]:
        """Returns list of seen OOV words."""
        return list(self._oov_words)

    def handle(self, word: str, embedding_fn) -> any:
        """
        Returns an embedding for an OOV word.

        Splits word into character trigrams, averages vectors of known trigrams.
        Returns None if no trigrams are known (caller should add perturbation).
        """
        if word in self._vocab:
            return embedding_fn(word)

        if word not in self._oov_words:
            self._oov_words.append(word)
            logger.debug("OOV word encountered: %s", word)

        trigrams = [word[i:i+3] for i in range(max(1, len(word) - 2))]
        vectors = [embedding_fn(t) for t in trigrams if t in self._vocab]

        if not vectors:
            return None

        import numpy as np
        return np.mean(vectors, axis=0)


class NLTKParser:
    """Parses text into a ParseNode tree using NLTK's RegexpParser."""

    def __init__(self, grammar: str = DEFAULT_GRAMMAR):
        self._chunker = nltk.RegexpParser(grammar)
        self._oov_handler: list[str] = []

    @property
    def oov_words(self) -> list[str]:
        """Returns words not recognized during POS tagging."""
        return list(self._oov_handler)

    def parse(self, text: str) -> ParseNode:
        """Tokenizes and parses text, returning a ParseNode tree."""
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        logger.debug("Tagged tokens: %s", tagged)
        chunked = self._chunker.parse(tagged)
        return _convert_tree(chunked)

    def from_tree(self, nltk_tree) -> ParseNode:
        """Converts an nltk.Tree to a ParseNode tree."""
        return _convert_tree(nltk_tree)

    def from_string(self, bracketed: str) -> ParseNode:
        """Parses a bracketed string into a ParseNode tree."""
        tree = nltk.Tree.fromstring(bracketed)
        return self.from_tree(tree)


def _convert_tree(tree) -> ParseNode:
    """Converts an nltk.Tree node or string leaf to a ParseNode."""
    if isinstance(tree, nltk.Tree):
        children = [_convert_tree(child) for child in tree]
        return ParseNode(label=tree.label(), children=children)
    else:
        word, pos = tree
        return ParseNode(label=pos, is_leaf=True, word=word, pos=pos)


def simplify(node: ParseNode) -> ParseNode:
    """Prunes single-child intermediate nodes (non-leaves) recursively."""
    if node.is_leaf:
        return node

    node.children = [simplify(child) for child in node.children]

    if len(node.children) == 1 and not node.children[0].is_leaf:
        return node.children[0]

    return node
