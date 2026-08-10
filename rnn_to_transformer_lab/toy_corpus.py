"""A toy English-to-Vietnamese parallel corpus, generated from a grammar.

Chapters 5 to 8 need a translation task, and the book set three constraints on
it that no public corpus satisfies at once: it has to finish on a CPU inside a
60-second per-experiment budget, it has to be reproducible from a seed with no
download, and the two languages have to differ in word order enough that
chapter 6's alignment matrix shows a crossing rather than a diagonal. So the
corpus is generated here rather than fetched, and the book says plainly that it
is a toy.

What is real about it is the one thing the chapters measure: **the noun phrase
reverses.** English puts the adjective before the noun and Vietnamese puts it
after, and English marks definiteness with a determiner while Vietnamese uses a
classifier that has no English word at all.

    the black cat  ->  con  meo  den
    a  new  lamp   ->  mot  cai  den  moi

A model that translates this position by position gets the adjective wrong on
every phrase that has one. That is the point: a diagonal alignment cannot solve
this corpus, so an alignment that does solve it has learned something.

What is *not* real about it, and no chapter may claim otherwise:

* The grammar is finite and context-free, so a model can learn it exactly.
  Nothing here measures translation quality in any sense a translator would
  recognise.
* There is no morphology, no agreement, no ambiguity and no rare words. The
  vocabulary is closed and every token appears thousands of times, so the
  out-of-vocabulary problem that penalised Sutskever et al.'s BLEU scores
  cannot arise here at all.
* Sentences are semantically silly ("the cat carries a new lamp"). The grammar
  constrains syntax, not sense.

The generator draws uniformly and independently, and the space is large enough
that a train and a test split drawn from different seeds barely overlap;
`disjoint_splits` removes what overlap there is rather than assuming there is
none.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

PAD, SOS, EOS = "<pad>", "<sos>", "<eos>"
SPECIALS = (PAD, SOS, EOS)

#: (english, vietnamese, classifier). The classifier is a Vietnamese word with
#: no English counterpart, which is one of the two reasons the target sentence
#: is usually longer than the source even though it says the same thing.
ANIMALS = (
    ("cat", "mèo", "con"),
    ("dog", "chó", "con"),
    ("bird", "chim", "con"),
    ("fish", "cá", "con"),
    ("horse", "ngựa", "con"),
)

OBJECTS = (
    ("book", "sách", "quyển"),
    ("chair", "ghế", "cái"),
    ("table", "bàn", "cái"),
    ("hat", "mũ", "cái"),
    ("lamp", "đèn", "cái"),
)

#: Adjectives an animal can take, and the wider set an object can take. Split
#: because "cũ" and "mới" do not apply to an animal in Vietnamese, and a
#: grammar that generated them would be teaching the model a string no speaker
#: would write.
ANIMAL_ADJECTIVES = (
    ("black", "đen"),
    ("white", "trắng"),
    ("small", "nhỏ"),
    ("big", "lớn"),
)
OBJECT_ADJECTIVES = ANIMAL_ADJECTIVES + (("old", "cũ"), ("new", "mới"))

#: A verb is one or two tokens on each side and they do not line up, so target
#: length is not a function of source length.
VERBS = (
    ("sees", ("nhìn", "thấy")),
    ("finds", ("tìm", "thấy")),
    ("wants", ("muốn",)),
    ("likes", ("thích",)),
    ("carries", ("mang",)),
    ("has", ("có",)),
)

#: Optional trailing phrase.
PLACES = (
    (("in", "the", "garden"), ("trong", "vườn")),
    (("on", "the", "table"), ("trên", "bàn")),
    (("at", "the", "window"), ("cạnh", "cửa", "sổ")),
)

AND = ("and", "và")
INDEFINITE = "một"


@dataclass(frozen=True)
class Pair:
    """One example, as token tuples rather than strings.

    Tuples so a pair is hashable and the split check below is a set operation
    rather than a quadratic scan.
    """

    source: tuple[str, ...]
    target: tuple[str, ...]


def _pick(generator: torch.Generator, n: int) -> int:
    return int(torch.randint(n, (1,), generator=generator).item())


def _noun_phrase(
    generator: torch.Generator, animals_only: bool
) -> tuple[list[str], list[str]]:
    """One noun phrase in both languages.

    These fifteen lines are the corpus. Everything else in this module is
    scaffolding around where the adjective goes: before the noun in English,
    after it in Vietnamese, with a classifier sitting where English has
    nothing at all.
    """
    table = ANIMALS if animals_only else ANIMALS + OBJECTS
    choice = _pick(generator, len(table))
    noun_en, noun_vi, classifier = table[choice]
    is_animal = animals_only or choice < len(ANIMALS)
    adjectives = ANIMAL_ADJECTIVES if is_animal else OBJECT_ADJECTIVES

    definite = _pick(generator, 2) == 0
    # Two thirds of phrases carry an adjective, so a model cannot score well by
    # ignoring the position question.
    has_adjective = _pick(generator, 3) != 0

    vietnamese = [] if definite else [INDEFINITE]
    vietnamese += [classifier, noun_vi]

    tail = []
    if has_adjective:
        adjective_en, adjective_vi = adjectives[_pick(generator, len(adjectives))]
        tail.append(adjective_en)
        vietnamese.append(adjective_vi)
    tail.append(noun_en)

    # "an old chair", not "a old chair". Vietnamese has no such agreement, so
    # this is one more place where a token on the source side answers to
    # nothing on the target side.
    determiner = "the" if definite else ("an" if tail[0][0] in "aeiou" else "a")
    return [determiner] + tail, vietnamese


def _clause(generator: torch.Generator) -> tuple[list[str], list[str]]:
    subject_en, subject_vi = _noun_phrase(generator, animals_only=True)
    verb_en, verb_vi = VERBS[_pick(generator, len(VERBS))]
    object_en, object_vi = _noun_phrase(generator, animals_only=False)

    english = subject_en + [verb_en] + object_en
    vietnamese = subject_vi + list(verb_vi) + object_vi

    if _pick(generator, 2) == 0:
        place_en, place_vi = PLACES[_pick(generator, len(PLACES))]
        english += list(place_en)
        vietnamese += list(place_vi)
    return english, vietnamese


def sentence(generator: torch.Generator, max_clauses: int = 2) -> Pair:
    """One sentence of one or two clauses, in both languages."""
    n_clauses = 1 + _pick(generator, max_clauses)
    english: list[str] = []
    vietnamese: list[str] = []
    for index in range(n_clauses):
        if index:
            english.append(AND[0])
            vietnamese.append(AND[1])
        clause_en, clause_vi = _clause(generator)
        english += clause_en
        vietnamese += clause_vi
    return Pair(source=tuple(english), target=tuple(vietnamese))


def corpus(n: int, seed: int, max_clauses: int = 2) -> list[Pair]:
    """`n` sentences drawn from the grammar, deterministic in `seed`."""
    generator = torch.Generator().manual_seed(seed)
    return [sentence(generator, max_clauses) for _ in range(n)]


def disjoint_splits(
    n_train: int, n_test: int, seed: int = 0, max_clauses: int = 2
) -> tuple[list[Pair], list[Pair]]:
    """A train and a test split with no source sentence in common.

    Two seeds drawn from the same distribution overlap in a handful of
    sentences at most, but "at most" is not a measurement, and a test set that
    quietly shares rows with training is the oldest way to report an accuracy
    that is not there. Oversample the test draw, drop anything already in
    train, then take the first `n_test`.
    """
    train = corpus(n_train, seed=seed, max_clauses=max_clauses)
    seen = {pair.source for pair in train}
    fresh = [
        pair
        for pair in corpus(n_test * 2, seed=seed + 9973, max_clauses=max_clauses)
        if pair.source not in seen
    ]
    if len(fresh) < n_test:
        raise ValueError(f"only {len(fresh)} unseen sentences, wanted {n_test}")
    return train, fresh[:n_test]


class Vocabulary:
    """Token to index, with the three special symbols pinned to 0, 1 and 2.

    Built from the grammar rather than from a sample, so the same indices come
    out whatever number of sentences an experiment draws. An experiment that
    built its vocabulary from its own training split would get a different
    layout at every size, and the ensemble experiment would then be averaging
    distributions over different alphabets.
    """

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = list(SPECIALS) + sorted(set(tokens) - set(SPECIALS))
        self.index = {token: i for i, token in enumerate(self.tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, tokens: tuple[str, ...] | list[str]) -> list[int]:
        return [self.index[token] for token in tokens]

    def decode(self, indices: list[int] | tuple[int, ...]) -> list[str]:
        return [self.tokens[i] for i in indices]


def vocabularies() -> tuple[Vocabulary, Vocabulary]:
    """The two closed vocabularies, enumerated from the grammar itself."""
    source: list[str] = ["the", "a", "an", AND[0]]
    target: list[str] = [INDEFINITE, AND[1]]
    for noun_en, noun_vi, classifier in ANIMALS + OBJECTS:
        source.append(noun_en)
        target += [noun_vi, classifier]
    for adjective_en, adjective_vi in OBJECT_ADJECTIVES:
        source.append(adjective_en)
        target.append(adjective_vi)
    for verb_en, verb_vi in VERBS:
        source.append(verb_en)
        target += list(verb_vi)
    for place_en, place_vi in PLACES:
        source += list(place_en)
        target += list(place_vi)
    return Vocabulary(source), Vocabulary(target)


def batches(
    pairs: list[Pair],
    source_vocab: Vocabulary,
    target_vocab: Vocabulary,
    batch_size: int,
    generator: torch.Generator,
    reverse_source: bool = False,
    sort_by_length: bool = True,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Pad into batches, shuffled, optionally with the source reversed.

    `sort_by_length` groups sentences of similar length into a batch before the
    batches themselves are shuffled. Sutskever et al. section 3.4 does the same
    thing and reports a 2x speedup; here it matters for the same reason, which
    is that a batch of mixed lengths is mostly padding and a padded position
    still costs a time step.

    Returns (source, target) with shape (steps, batch). The target carries
    <sos> at the front and <eos> at the back, so a decoder reads `target[:-1]`
    and is scored against `target[1:]`.
    """
    order = list(range(len(pairs)))
    if sort_by_length:
        order.sort(key=lambda i: len(pairs[i].source))

    grouped = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    shuffle = torch.randperm(len(grouped), generator=generator).tolist()

    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for position in shuffle:
        rows = grouped[position]
        sources = []
        targets = []
        for i in rows:
            tokens = list(pairs[i].source)
            if reverse_source:
                tokens.reverse()
            sources.append(source_vocab.encode(tokens))
            targets.append(
                [target_vocab.index[SOS]]
                + target_vocab.encode(pairs[i].target)
                + [target_vocab.index[EOS]]
            )
        out.append((_pad(sources), _pad(targets)))
    return out


def _pad(rows: list[list[int]]) -> torch.Tensor:
    width = max(len(row) for row in rows)
    padded = torch.zeros(width, len(rows), dtype=torch.long)  # <pad> is index 0
    for column, row in enumerate(rows):
        padded[: len(row), column] = torch.tensor(row, dtype=torch.long)
    return padded


def statistics(pairs: list[Pair]) -> dict[str, float]:
    """Lengths and shape counts, for the table chapter 5 prints."""
    source_lengths = [len(pair.source) for pair in pairs]
    target_lengths = [len(pair.target) for pair in pairs]
    n = len(pairs)
    return {
        "pairs": n,
        "src_min": min(source_lengths),
        "src_max": max(source_lengths),
        "src_mean": sum(source_lengths) / n,
        "tgt_min": min(target_lengths),
        "tgt_max": max(target_lengths),
        "tgt_mean": sum(target_lengths) / n,
        "longer_target": sum(t > s for s, t in zip(source_lengths, target_lengths)) / n,
    }
