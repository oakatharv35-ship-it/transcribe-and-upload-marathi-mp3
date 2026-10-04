import argparse
import csv
import html
import json
import re
import unicodedata

from collections import defaultdict, Counter
from pathlib import Path

from rapidfuzz import fuzz, process


# ============================================================
# MATCH CONFIGURATION
# ============================================================

MATCH_THRESHOLD = 82.0

# 2-word fuzzy anchor:
# both words must be > 3 characters.
BIGRAM_MATCH_THRESHOLD = 88.0

# 3-word fuzzy anchor:
# no minimum word length.
TRIGRAM_MATCH_THRESHOLD = 76.0

MIN_PAIR_WORD_CHARS = 3

MIN_MATCH_WORDS = 2
MAX_MATCH_WORDS = 16

MAX_NGRAM_CANDIDATES = 4

MAX_CANDIDATE_STARTS_PER_VERSE = 12

POSITION_SLACK = 2

# Prevent one random 2-word fuzzy match from
# creating a relation.
MIN_ANCHOR_VOTES = 5.0

# Maximum different verses for exactly the
# same detected timestamp interval.
MAX_MATCHES_PER_TIME_REGION = 2


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_marathi(text: str) -> str:

    text = unicodedata.normalize(
        "NFC",
        text
    )

    text = html.unescape(text)

    # Remove generated tags if processing a file again.
    text = re.sub(
        r"</?ovi\b[^>]*>",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"<relations>.*?</relations>",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Remove labels such as:
    #
    # [Adhyay 11, Ovi 258]
    #
    text = re.sub(
        r"\[\s*Adhyay\s+\d+\s*,\s*Ovi\s+\d+[^\]]*\]",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = text.replace(
        "🏷️",
        " "
    )

    # Markdown formatting.
    text = text.replace(
        "**",
        " "
    )

    text = text.replace(
        "__",
        " "
    )

    text = text.replace(
        "*",
        " "
    )

    text = text.replace(
        "_",
        " "
    )

    # Zero-width characters.
    text = re.sub(
        r"[\u200b-\u200d\ufeff]",
        "",
        text
    )

    # English + Devanagari numbers.
    text = re.sub(
        r"[0-9०-९]+",
        " ",
        text
    )

    # Punctuation.
    text = re.sub(
        r"""[
            \.,!?;:'"“”‘’
            \(\)\[\]\{\}<>
            ।॥\-–—…:/\\|
        ]""",
        " ",
        text,
        flags=re.VERBOSE
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip().lower()


# ============================================================
# WORD RUNS
# ============================================================

def get_bigrams(words):

    runs = []

    for i in range(
        len(words) - 1
    ):

        w1 = words[i]
        w2 = words[i + 1]

        if (
            len(w1) > MIN_PAIR_WORD_CHARS
            and
            len(w2) > MIN_PAIR_WORD_CHARS
        ):

            runs.append({
                "text":
                    f"{w1} {w2}",

                "position":
                    i
            })

    return runs


def get_trigrams(words):

    runs = []

    for i in range(
        len(words) - 2
    ):

        runs.append({
            "text":
                " ".join(
                    words[i:i + 3]
                ),

            "position":
                i
        })

    return runs


# ============================================================
# LOAD DNYANESHWARI
# ============================================================

def load_ovis(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if "verses" not in data:

        raise ValueError(
            "Dnyaneshwari JSON must contain "
            "a top-level 'verses' array."
        )

    ovis = []

    for row in data["verses"]:

        verse_id = row.get(
            "verse_id"
        )

        chapter = row.get(
            "chapter_number"
        )

        verse = row.get(
            "verse_number"
        )

        text = row.get(
            "shloka_devanagari"
        )

        if (
            not verse_id
            or chapter is None
            or verse is None
            or not text
        ):
            continue

        normalized = normalize_marathi(
            text
        )

        words = normalized.split()

        if len(words) < 2:
            continue

        ovis.append({
            "id":
                verse_id,

            "book":
                "Dnyaneshwari",

            "chapter":
                chapter,

            "verse":
                verse,

            "text":
                text,

            "normalized":
                normalized,

            "words":
                words,

            "bigrams":
                get_bigrams(
                    words
                ),

            "trigrams":
                get_trigrams(
                    words
                ),
        })

    print(
        f"Loaded {len(ovis)} "
        f"Dnyaneshwari verses"
    )

    return ovis


# ============================================================
# BUILD FUZZY N-GRAM INDEX
# ============================================================

def build_fuzzy_ngram_index(
    ovis
):

    bigram_lookup = defaultdict(
        list
    )

    trigram_lookup = defaultdict(
        list
    )

    for verse_index, ovi in enumerate(
        ovis
    ):

        for run in ovi[
            "bigrams"
        ]:

            bigram_lookup[
                run["text"]
            ].append({
                "verse_index":
                    verse_index,

                "verse_position":
                    run["position"]
            })

        for run in ovi[
            "trigrams"
        ]:

            trigram_lookup[
                run["text"]
            ].append({
                "verse_index":
                    verse_index,

                "verse_position":
                    run["position"]
            })

    bigram_choices = list(
        bigram_lookup.keys()
    )

    trigram_choices = list(
        trigram_lookup.keys()
    )

    print(
        f"Unique bigrams: "
        f"{len(bigram_choices):,}"
    )

    print(
        f"Unique trigrams: "
        f"{len(trigram_choices):,}"
    )

    return {
        "bigram_lookup":
            bigram_lookup,

        "trigram_lookup":
            trigram_lookup,

        "bigram_choices":
            bigram_choices,

        "trigram_choices":
            trigram_choices,
    }


# ============================================================
# TRANSCRIPT PARSING
# ============================================================

TIMESTAMP_PATTERN = re.compile(
    r"^\s*\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.*?)\s*$"
)

HEADER_PATTERN = re.compile(
    r"^\s*#{1,6}\s+"
)


def clean_original_text(
    text
):

    text = html.unescape(
        text
    )

    text = re.sub(
        r"</?ovi\b[^>]*>",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"<relations>.*?</relations>",
        "",
        text,
        flags=re.IGNORECASE
    )

    return text.strip()


def clean_md_file_lines(
    raw_lines
):
    """
    Remove:

    1. Everything before the first timestamped line.
       This removes sections such as:

           ### 1. SCRIPTURAL CONTEXT & METADATA
           ...
           ### 2. VERBATIM MARATHI TRANSCRIPT

    2. Markdown headers beginning with:
           #
           ##
           ###
           etc.

    3. Horizontal separators such as:
           ---
           ***
           ___

    The returned lines therefore contain only
    the actual transcript/body.
    """

    first_timestamp_index = None

    # Find first actual transcript line.
    for i, line in enumerate(
        raw_lines
    ):

        if TIMESTAMP_PATTERN.match(
            line.strip()
        ):

            first_timestamp_index = i
            break

    if first_timestamp_index is None:

        return []

    # Remove everything before transcript.
    lines = raw_lines[
        first_timestamp_index:
    ]

    cleaned = []

    for line in lines:

        stripped = line.strip()

        # Remove Markdown headers.
        if HEADER_PATTERN.match(
            stripped
        ):
            continue

        # Remove horizontal separators.
        if stripped in {
            "---",
            "***",
            "___"
        }:
            continue

        # Ignore empty lines.
        if not stripped:
            continue

        cleaned.append(
            line
        )

    return cleaned


def parse_transcript(
    path
):
    """
    Timestamps are removed for matching.

    Example:

        [00:00] hello
        [00:05] world

    becomes:

        hello world

    Position information is retained so matches can
    later be mapped back to timestamps.
    """

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        original_raw_lines = (
            f.readlines()
        )

    # ========================================================
    # REMOVE FIRST SECTION + HEADERS
    # ========================================================

    raw_lines = (
        clean_md_file_lines(
            original_raw_lines
        )
    )

    transcript_lines = []

    continuous_parts = []

    current_char = 0

    for raw_index, raw_line in enumerate(
        raw_lines
    ):

        original = (
            raw_line.rstrip(
                "\r\n"
            )
        )

        match = (
            TIMESTAMP_PATTERN.match(
                original
            )
        )

        # Only timestamped transcript lines are matched.
        if not match:
            continue

        timestamp = (
            match.group(1)
        )

        original_text = (
            clean_original_text(
                match.group(2)
            )
        )

        normalized = (
            normalize_marathi(
                original_text
            )
        )

        if not normalized:
            continue

        if continuous_parts:

            current_char += 1

        start_char = (
            current_char
        )

        continuous_parts.append(
            normalized
        )

        current_char += len(
            normalized
        )

        end_char = (
            current_char
        )

        transcript_lines.append({
            "raw_index":
                raw_index,

            "timestamp":
                timestamp,

            "text":
                original_text,

            "normalized":
                normalized,

            "start_char":
                start_char,

            "end_char":
                end_char,
        })

    continuous_text = (
        " ".join(
            continuous_parts
        )
    )

    return (
        raw_lines,
        transcript_lines,
        continuous_text
    )


# ============================================================
# TOKENIZE
# ============================================================

def tokenize_continuous_text(
    text
):

    tokens = []

    for match in re.finditer(
        r"\S+",
        text
    ):

        tokens.append({
            "word":
                match.group(),

            "start_char":
                match.start(),

            "end_char":
                match.end()
        })

    return tokens


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def collect_candidate_starts(
    transcript_tokens,
    fuzzy_index
):

    words = [
        token["word"]
        for token in transcript_tokens
    ]

    candidate_starts = (
        defaultdict(
            Counter
        )
    )

    # ========================================================
    # THREE-WORD FUZZY ANCHORS
    # ========================================================

    for transcript_position in range(
        len(words) - 2
    ):

        phrase = " ".join(
            words[
                transcript_position:
                transcript_position + 3
            ]
        )

        results = process.extract(
            phrase,

            fuzzy_index[
                "trigram_choices"
            ],

            scorer=fuzz.ratio,

            score_cutoff=
                TRIGRAM_MATCH_THRESHOLD,

            limit=
                MAX_NGRAM_CANDIDATES
        )

        for (
            matched_phrase,
            score,
            _
        ) in results:

            locations = (
                fuzzy_index[
                    "trigram_lookup"
                ][matched_phrase]
            )

            for location in locations:

                verse_index = (
                    location[
                        "verse_index"
                    ]
                )

                verse_position = (
                    location[
                        "verse_position"
                    ]
                )

                estimated_start = (
                    transcript_position
                    - verse_position
                )

                weight = (
                    4.0
                    +
                    score / 100.0
                )

                candidate_starts[
                    verse_index
                ][
                    estimated_start
                ] += weight

    # ========================================================
    # TWO-WORD FUZZY ANCHORS
    # ========================================================

    for transcript_position in range(
        len(words) - 1
    ):

        w1 = words[
            transcript_position
        ]

        w2 = words[
            transcript_position + 1
        ]

        if (
            len(w1)
            <= MIN_PAIR_WORD_CHARS

            or

            len(w2)
            <= MIN_PAIR_WORD_CHARS
        ):
            continue

        phrase = (
            f"{w1} {w2}"
        )

        results = process.extract(
            phrase,

            fuzzy_index[
                "bigram_choices"
            ],

            scorer=fuzz.ratio,

            score_cutoff=
                BIGRAM_MATCH_THRESHOLD,

            limit=
                MAX_NGRAM_CANDIDATES
        )

        for (
            matched_phrase,
            score,
            _
        ) in results:

            locations = (
                fuzzy_index[
                    "bigram_lookup"
                ][matched_phrase]
            )

            for location in locations:

                verse_index = (
                    location[
                        "verse_index"
                    ]
                )

                verse_position = (
                    location[
                        "verse_position"
                    ]
                )

                estimated_start = (
                    transcript_position
                    - verse_position
                )

                weight = (
                    2.0
                    +
                    score / 100.0
                )

                candidate_starts[
                    verse_index
                ][
                    estimated_start
                ] += weight

    # ========================================================
    # FILTER WEAK CANDIDATES
    # ========================================================

    filtered = defaultdict(
        Counter
    )

    for (
        verse_index,
        starts
    ) in candidate_starts.items():

        for (
            start,
            votes
        ) in starts.items():

            if (
                votes
                >= MIN_ANCHOR_VOTES
            ):

                filtered[
                    verse_index
                ][
                    start
                ] = votes

    return filtered


# ============================================================
# WORD-RUN VERIFICATION
# ============================================================

def best_run_score(
    transcript_words,
    verse_words
):

    best = None

    # ========================================================
    # TWO-WORD MATCH
    # ========================================================

    for ti in range(
        len(transcript_words) - 1
    ):

        t1 = transcript_words[ti]
        t2 = transcript_words[
            ti + 1
        ]

        if (
            len(t1)
            <= MIN_PAIR_WORD_CHARS

            or

            len(t2)
            <= MIN_PAIR_WORD_CHARS
        ):
            continue

        transcript_phrase = (
            f"{t1} {t2}"
        )

        for vi in range(
            len(verse_words) - 1
        ):

            v1 = verse_words[vi]
            v2 = verse_words[
                vi + 1
            ]

            if (
                len(v1)
                <= MIN_PAIR_WORD_CHARS

                or

                len(v2)
                <= MIN_PAIR_WORD_CHARS
            ):
                continue

            verse_phrase = (
                f"{v1} {v2}"
            )

            score = fuzz.ratio(
                transcript_phrase,
                verse_phrase
            )

            if (
                score
                >= BIGRAM_MATCH_THRESHOLD
            ):

                if (
                    best is None
                    or score > best
                ):

                    best = score

    # ========================================================
    # THREE-WORD MATCH
    # ========================================================

    for ti in range(
        len(transcript_words) - 2
    ):

        transcript_phrase = (
            " ".join(
                transcript_words[
                    ti:ti + 3
                ]
            )
        )

        for vi in range(
            len(verse_words) - 2
        ):

            verse_phrase = (
                " ".join(
                    verse_words[
                        vi:vi + 3
                    ]
                )
            )

            score = fuzz.ratio(
                transcript_phrase,
                verse_phrase
            )

            if (
                score
                >= TRIGRAM_MATCH_THRESHOLD
            ):

                if (
                    best is None
                    or score > best
                ):

                    best = score

    return best


# ============================================================
# EVALUATE CANDIDATE
# ============================================================

def evaluate_candidate(
    verse,
    estimated_start,
    transcript_tokens
):

    verse_words = (
        verse["words"]
    )

    verse_length = len(
        verse_words
    )

    max_length = min(
        MAX_MATCH_WORDS,
        verse_length + 4
    )

    best = None

    for shift in range(
        -POSITION_SLACK,
        POSITION_SLACK + 1
    ):

        start = (
            estimated_start
            + shift
        )

        if start < 0:
            continue

        if (
            start
            >= len(
                transcript_tokens
            )
        ):
            continue

        for window_length in range(
            MIN_MATCH_WORDS,
            max_length + 1
        ):

            end = (
                start
                + window_length
            )

            if (
                end
                > len(
                    transcript_tokens
                )
            ):
                break

            transcript_words = [
                token["word"]

                for token
                in transcript_tokens[
                    start:end
                ]
            ]

            transcript_text = (
                " ".join(
                    transcript_words
                )
            )

            partial_score = (
                fuzz.partial_ratio(
                    transcript_text,
                    verse[
                        "normalized"
                    ]
                )
            )

            token_score = (
                fuzz.token_set_ratio(
                    transcript_text,
                    verse[
                        "normalized"
                    ]
                )
            )

            if (
                partial_score
                >= token_score
            ):

                text_score = (
                    partial_score
                )

                method = (
                    "partial"
                )

            else:

                text_score = (
                    token_score
                )

                method = (
                    "token_set"
                )

            count = len(
                transcript_words
            )

            # =================================================
            # SHORTER MATCHES REQUIRE HIGHER CONFIDENCE
            # =================================================

            if count == 2:

                required_score = 90.0

            elif count == 3:

                required_score = 84.0

            else:

                required_score = (
                    MATCH_THRESHOLD
                )

            if (
                text_score
                < required_score
            ):
                continue

            run_score = (
                best_run_score(
                    transcript_words,
                    verse_words
                )
            )

            if (
                run_score is None
            ):
                continue

            final_score = (
                text_score * 0.55
                +
                run_score * 0.45
            )

            candidate = {
                "start_word":
                    start,

                "end_word":
                    end - 1,

                "score":
                    round(
                        final_score,
                        2
                    ),

                "text_score":
                    round(
                        text_score,
                        2
                    ),

                "run_score":
                    round(
                        run_score,
                        2
                    ),

                "method":
                    method,

                "verse":
                    verse
            }

            if (
                best is None
                or
                candidate["score"]
                > best["score"]
            ):

                best = candidate

    return best


# ============================================================
# FIND MATCHES
# ============================================================

def find_matches(
    ovis,
    transcript_tokens,
    candidate_starts
):

    matches = []

    for (
        verse_index,
        start_counter
    ) in candidate_starts.items():

        verse = ovis[
            verse_index
        ]

        strongest_positions = (
            start_counter.most_common(
                MAX_CANDIDATE_STARTS_PER_VERSE
            )
        )

        for (
            estimated_start,
            votes
        ) in strongest_positions:

            result = (
                evaluate_candidate(
                    verse,
                    estimated_start,
                    transcript_tokens
                )
            )

            if result is None:
                continue

            result[
                "anchor_votes"
            ] = round(
                votes,
                2
            )

            matches.append(
                result
            )

    return matches


# ============================================================
# DEDUPLICATE
# ============================================================

def ranges_overlap(
    a,
    b
):

    return not (
        a["end_word"]
        < b["start_word"]

        or

        b["end_word"]
        < a["start_word"]
    )


def deduplicate_matches(
    matches
):

    matches.sort(
        key=lambda x: (
            x["score"],
            x.get(
                "anchor_votes",
                0
            ),
            x["run_score"]
        ),
        reverse=True
    )

    selected = []

    for candidate in matches:

        duplicate = False

        for existing in selected:

            # Different verses are okay.
            if (
                candidate[
                    "verse"
                ]["id"]

                !=

                existing[
                    "verse"
                ]["id"]
            ):
                continue

            if ranges_overlap(
                candidate,
                existing
            ):

                duplicate = True
                break

        if not duplicate:

            selected.append(
                candidate
            )

    selected.sort(
        key=lambda x:
            x["start_word"]
    )

    return selected


# ============================================================
# WORD -> CHARACTER POSITION
# ============================================================

def add_character_spans(
    matches,
    transcript_tokens
):

    for match in matches:

        first = (
            transcript_tokens[
                match[
                    "start_word"
                ]
            ]
        )

        last = (
            transcript_tokens[
                match[
                    "end_word"
                ]
            ]
        )

        match[
            "start_char"
        ] = first[
            "start_char"
        ]

        match[
            "end_char"
        ] = last[
            "end_char"
        ]

    return matches


# ============================================================
# CHARACTER POSITION -> TIMESTAMP
# ============================================================

def assign_timestamps(
    matches,
    transcript_lines
):

    for match in matches:

        overlapping = []

        for i, line in enumerate(
            transcript_lines
        ):

            overlap = not (
                line[
                    "end_char"
                ]
                <=
                match[
                    "start_char"
                ]

                or

                line[
                    "start_char"
                ]
                >=
                match[
                    "end_char"
                ]
            )

            if overlap:

                overlapping.append(
                    i
                )

        if not overlapping:
            continue

        first_index = (
            overlapping[0]
        )

        last_index = (
            overlapping[-1]
        )

        match[
            "start_line"
        ] = first_index

        match[
            "end_line"
        ] = last_index

        match[
            "start_timestamp"
        ] = (
            transcript_lines[
                first_index
            ][
                "timestamp"
            ]
        )

        if (
            last_index + 1
            < len(
                transcript_lines
            )
        ):

            match[
                "end_timestamp"
            ] = (
                transcript_lines[
                    last_index + 1
                ][
                    "timestamp"
                ]
            )

        else:

            match[
                "end_timestamp"
            ] = (
                transcript_lines[
                    last_index
                ][
                    "timestamp"
                ]
            )

    return matches


# ============================================================
# LIMIT MATCHES FOR SAME TIME REGION
# ============================================================

def limit_matches_per_region(
    matches
):

    grouped = defaultdict(
        list
    )

    for match in matches:

        if (
            "start_timestamp"
            not in match
        ):
            continue

        key = (
            match[
                "start_timestamp"
            ],

            match[
                "end_timestamp"
            ]
        )

        grouped[
            key
        ].append(
            match
        )

    final = []

    for group in grouped.values():

        group.sort(
            key=lambda x: (
                x["score"],
                x.get(
                    "anchor_votes",
                    0
                ),
                x["run_score"]
            ),
            reverse=True
        )

        final.extend(
            group[
                :
                MAX_MATCHES_PER_TIME_REGION
            ]
        )

    final.sort(
        key=lambda x:
            x["start_word"]
    )

    return final


# ============================================================
# TAG/CLEAN MARKDOWN
# ============================================================

def tag_markdown(
    raw_lines,
    transcript_lines,
    matches
):
    """
    Output MD deliberately contains ONLY transcript lines.

    This means:
      - Metadata section is gone
      - Executive summary is gone
      - Markdown headers are gone
      - Only timestamped transcript remains
    """

    relation_lookup = (
        defaultdict(
            list
        )
    )

    for match in matches:

        if (
            "start_line"
            not in match
        ):
            continue

        for line_index in range(
            match["start_line"],
            match["end_line"] + 1
        ):

            relation_lookup[
                line_index
            ].append(
                match
            )

    output = []

    # transcript_lines contains only timestamped body.
    for line_index, line in enumerate(
        transcript_lines
    ):

        output_line = (
            f'[{line["timestamp"]}] '
            f'{line["text"]}'
        )

        related = (
            relation_lookup.get(
                line_index,
                []
            )
        )

        if related:

            tags = []
            seen = set()

            for match in related:

                verse = (
                    match["verse"]
                )

                if (
                    verse["id"]
                    in seen
                ):
                    continue

                seen.add(
                    verse["id"]
                )

                tags.append(
                    (
                        f'{verse["id"]}'
                        f' '
                        f'(score='
                        f'{match["score"]})'
                    )
                )

            output_line += (
                " <relations>"
                +
                "; ".join(
                    tags
                )
                +
                "</relations>"
            )

        output.append(
            output_line + "\n"
        )

    return output


# ============================================================
# CSV ROW CREATION
# ============================================================

def create_csv_rows(
    transcript_path,
    matches
):

    filename = (
        Path(
            transcript_path
        ).name
    )

    rows = []

    for match in matches:

        verse = (
            match["verse"]
        )

        rows.append({
            "File name":
                filename,

            "Related Book":
                verse[
                    "book"
                ],

            "Chapter and Verse":
                (
                    f'Chapter '
                    f'{verse["chapter"]}, '
                    f'Verse '
                    f'{verse["verse"]}'
                ),

            "Time Stamp":
                (
                    f'{match["start_timestamp"]}'
                    f' - '
                    f'{match["end_timestamp"]}'
                )
        })

    return rows


# ============================================================
# WRITE ONE COMBINED CSV
# ============================================================

def write_combined_csv(
    output_path,
    rows
):

    with open(
        output_path,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "File name",
                "Related Book",
                "Chapter and Verse",
                "Time Stamp"
            ]
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# AUDIO ID
# ============================================================

def safe_audio_id(
    path
):

    stem = (
        Path(
            path
        ).stem.lower()
    )

    stem = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        stem
    ).strip(
        "_"
    )

    if not stem:

        stem = (
            "transcript"
        )

    return (
        f"audio_{stem}"
    )


# ============================================================
# CREATE CONNECTION EDGES
# ============================================================

def create_audio_edges(
    transcript_path,
    matches
):

    audio_id = (
        safe_audio_id(
            transcript_path
        )
    )

    edges = []

    for index, match in enumerate(
        matches,
        start=1
    ):

        verse = (
            match[
                "verse"
            ]
        )

        # Timestamp makes clip IDs more stable than only
        # sequential numbering.
        timestamp_slug = re.sub(
            r"[^0-9]+",
            "_",
            match[
                "start_timestamp"
            ]
        ).strip(
            "_"
        )

        clip_id = (
            f"{audio_id}"
            f"_"
            f"{timestamp_slug}"
            f"_"
            f"{index:03d}"
        )

        edges.append({
            "edge_id":
                (
                    f'{verse["id"]}'
                    f'__'
                    f'{clip_id}'
                ),

            "source_id":
                verse["id"],

            "target_id":
                clip_id,

            "relation":
                "EXPLAINED_BY",

            "audio_id":
                audio_id,

            "start_time":
                match[
                    "start_timestamp"
                ],

            "end_time":
                match[
                    "end_timestamp"
                ],

            "match_type":
                match[
                    "method"
                ].upper(),

            "match_score":
                match[
                    "score"
                ],

            "text_score":
                match[
                    "text_score"
                ],

            "word_run_score":
                match[
                    "run_score"
                ],

            "anchor_votes":
                match[
                    "anchor_votes"
                ]
        })

    return edges


# ============================================================
# CONNECTIONS.JSON
# ============================================================

def load_connections(
    path
):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = (
            json.load(
                f
            )
        )

    if isinstance(
        data,
        list
    ):

        return (
            data,
            data
        )

    if (
        isinstance(
            data,
            dict
        )
        and
        "edges" in data
    ):

        return (
            data,
            data[
                "edges"
            ]
        )

    raise ValueError(
        "connections.json must be "
        "an array or contain an "
        "'edges' array."
    )


def update_connections_in_memory(
    edges,
    new_edges
):

    existing_ids = {
        edge.get(
            "edge_id"
        )
        for edge in edges
    }

    added = 0

    for edge in new_edges:

        if (
            edge[
                "edge_id"
            ]
            in existing_ids
        ):
            continue

        edges.append(
            edge
        )

        existing_ids.add(
            edge[
                "edge_id"
            ]
        )

        added += 1

    return added


def save_connections(
    path,
    data
):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(
    transcript_path,
    output_md_path,
    ovis,
    fuzzy_index
):

    print()
    print("=" * 90)
    print(
        f"Processing: "
        f"{transcript_path.name}"
    )
    print("=" * 90)

    (
        raw_lines,
        transcript_lines,
        continuous_text
    ) = parse_transcript(
        transcript_path
    )

    if not transcript_lines:

        print(
            "No timestamped transcript "
            "lines found. Skipping."
        )

        return [], []

    print(
        f"Transcript lines: "
        f"{len(transcript_lines):,}"
    )

    transcript_tokens = (
        tokenize_continuous_text(
            continuous_text
        )
    )

    print(
        f"Words: "
        f"{len(transcript_tokens):,}"
    )

    candidate_starts = (
        collect_candidate_starts(
            transcript_tokens,
            fuzzy_index
        )
    )

    print(
        f"Candidate verses: "
        f"{len(candidate_starts):,}"
    )

    matches = find_matches(
        ovis,
        transcript_tokens,
        candidate_starts
    )

    print(
        f"Raw matches: "
        f"{len(matches):,}"
    )

    matches = (
        deduplicate_matches(
            matches
        )
    )

    matches = (
        add_character_spans(
            matches,
            transcript_tokens
        )
    )

    matches = (
        assign_timestamps(
            matches,
            transcript_lines
        )
    )

    matches = (
        limit_matches_per_region(
            matches
        )
    )

    print(
        f"Final relations: "
        f"{len(matches):,}"
    )

    # ========================================================
    # WRITE CLEANED + TAGGED MD
    # ========================================================

    tagged_md = tag_markdown(
        raw_lines,
        transcript_lines,
        matches
    )

    output_md_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_md_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.writelines(
            tagged_md
        )

    csv_rows = (
        create_csv_rows(
            transcript_path,
            matches
        )
    )

    edges = (
        create_audio_edges(
            transcript_path,
            matches
        )
    )

    return (
        csv_rows,
        edges
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    # ========================================================
    # INPUT DIRECTORY
    # ========================================================

    parser.add_argument(
        "--transcript-dir",
        required=True,
        help=(
            "Directory containing "
            "all transcript .md files"
        )
    )

    parser.add_argument(
        "--dnyaneshwari",
        required=True
    )

    parser.add_argument(
        "--connections",
        required=True
    )

    # ========================================================
    # OUTPUTS
    # ========================================================

    parser.add_argument(
        "--output-md-dir",
        required=True,
        help=(
            "Directory for cleaned/tagged "
            "Markdown files"
        )
    )

    parser.add_argument(
        "--output-connections",
        required=True
    )

    parser.add_argument(
        "--output-csv",
        required=True,
        help=(
            "One combined CSV containing "
            "relations from every transcript"
        )
    )

    args = parser.parse_args()

    transcript_dir = Path(
        args.transcript_dir
    )

    output_md_dir = Path(
        args.output_md_dir
    )

    # ========================================================
    # VALIDATE DIRECTORY
    # ========================================================

    if not transcript_dir.exists():

        raise FileNotFoundError(
            f"Transcript directory "
            f"does not exist: "
            f"{transcript_dir}"
        )

    if not transcript_dir.is_dir():

        raise ValueError(
            f"--transcript-dir must "
            f"be a directory: "
            f"{transcript_dir}"
        )

    output_md_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # FIND ALL .MD FILES
    # ========================================================

    transcript_files = sorted(
        transcript_dir.glob(
            "*.md"
        )
    )

    if not transcript_files:

        raise RuntimeError(
            f"No .md files found in "
            f"{transcript_dir}"
        )

    print(
        f"Found "
        f"{len(transcript_files)} "
        f"Markdown files."
    )

    # ========================================================
    # LOAD DNYANESHWARI ONCE
    # ========================================================

    ovis = load_ovis(
        args.dnyaneshwari
    )

    if not ovis:

        raise RuntimeError(
            "No Dnyaneshwari verses loaded."
        )

    # ========================================================
    # BUILD INDEX ONCE
    # ========================================================

    fuzzy_index = (
        build_fuzzy_ngram_index(
            ovis
        )
    )

    # ========================================================
    # LOAD CONNECTIONS ONCE
    # ========================================================

    (
        connection_data,
        connection_edges
    ) = load_connections(
        args.connections
    )

    # ========================================================
    # PROCESS EVERY FILE
    # ========================================================

    all_csv_rows = []

    total_connections_added = 0

    processed = 0

    skipped = 0

    for transcript_path in transcript_files:

        # Preserve same filename in output dir.
        output_md_path = (
            output_md_dir
            /
            transcript_path.name
        )

        csv_rows, new_edges = (
            process_file(
                transcript_path=
                    transcript_path,

                output_md_path=
                    output_md_path,

                ovis=
                    ovis,

                fuzzy_index=
                    fuzzy_index
            )
        )

        if not csv_rows and not new_edges:

            skipped += 1
            continue

        all_csv_rows.extend(
            csv_rows
        )

        added = (
            update_connections_in_memory(
                connection_edges,
                new_edges
            )
        )

        total_connections_added += (
            added
        )

        processed += 1

    # ========================================================
    # WRITE ONE COMBINED CSV
    # ========================================================

    Path(
        args.output_csv
    ).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    write_combined_csv(
        args.output_csv,
        all_csv_rows
    )

    # ========================================================
    # SAVE CONNECTIONS ONCE
    # ========================================================

    Path(
        args.output_connections
    ).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    save_connections(
        args.output_connections,
        connection_data
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 90)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 90)

    print(
        f"Markdown files found: "
        f"{len(transcript_files)}"
    )

    print(
        f"Files processed: "
        f"{processed}"
    )

    print(
        f"Files skipped: "
        f"{skipped}"
    )

    print(
        f"Total CSV relations: "
        f"{len(all_csv_rows)}"
    )

    print(
        f"Connections added: "
        f"{total_connections_added}"
    )

    print(
        f"Combined CSV: "
        f"{args.output_csv}"
    )

    print(
        f"Cleaned Markdown directory: "
        f"{args.output_md_dir}"
    )

    print(
        f"Connections JSON: "
        f"{args.output_connections}"
    )


if __name__ == "__main__":
    main()