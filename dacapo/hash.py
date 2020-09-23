from .hash_phrase import hash_phrase

from importlib import resources as pkg_resources

from . import hash_words


NOUN_WORDLIST = pkg_resources.read_text(hash_words, "animals.txt").strip().split()
ADJECTIVE_WORDLIST = pkg_resources.read_text(hash_words, "colors.txt").strip().split()


def hash_adjective(token):
    return hash_phrase(token.encode(), dictionary=ADJECTIVE_WORDLIST)


def hash_noun(token):
    return hash_phrase(token.encode(), dictionary=NOUN_WORDLIST)


def human_readable_hash(tokens):
    """Create a human readable hash for the given tokens."""

    human_hash = "-".join(
        [
            hash_phrase(token.encode(), dictionary=ADJECTIVE_WORDLIST)
            for token in tokens[:-1]
        ]
    )
    human_hash = "-".join(
        [human_hash, hash_phrase(tokens[-1].encode(), dictionary=NOUN_WORDLIST)]
    )

    return human_hash
