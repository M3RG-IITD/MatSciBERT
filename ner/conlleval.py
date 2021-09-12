# Based on the script from https://github.com/spyysalo/conlleval.py/blob/master/conlleval.py
# 
# with the following license
# 
# The MIT License (MIT)
# 
# Copyright (c) 2016 Sampo Pyysalo
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Small changes were done to the original version, e.g. replacing the 
# sys.args format with function calls.

# Python version of the evaluation script from CoNLL'00-

# Intentional differences:
# - accept any space as delimiter by default
# - optional file argument (default STDIN)
# - option to set boundary (-b argument)
# - LaTeX output (-l argument) not supported
# - raw tags (-r argument) not supported

import re

from collections import defaultdict, namedtuple

ANY_SPACE = '<SPACE>'


class FormatError(Exception):
    pass


Metrics = namedtuple('Metrics', 'tp fp fn prec rec fscore')


class EvalCounts(object):
    def __init__(self):
        self.correct_chunk = 0    # number of correctly identified chunks
        self.correct_tags = 0     # number of correct chunk tags
        self.found_correct = 0    # number of chunks in corpus
        self.found_guessed = 0    # number of identified chunks
        self.token_counter = 0    # token counter (ignores sentence breaks)

        # counts by type
        self.t_correct_chunk = defaultdict(int)
        self.t_found_correct = defaultdict(int)
        self.t_found_guessed = defaultdict(int)


def parse_tag(t):
    m = re.match(r'^([^-]*)-(.*)$', t)
    return m.groups() if m else (t, '')


def evaluate(iterable, tmp_options={}):
    """
    :param iterable: for instance CSV file in BIO format, last two columns are gold and predicted labels respectively,
    can of course be a different type of iterable as well in which this holds!
    :param tmp_options:
    :return: counts for values
    """

    options = {'delimiter': tmp_options['delimiter'] if 'delimiter' in tmp_options else ANY_SPACE,
               'boundary': tmp_options['boundary'] if 'boundary' in tmp_options else '-X-',
               'otag': tmp_options['otag'] if 'otag' in tmp_options else 'O'}

    counts = EvalCounts()
    num_features = None       # number of features per line
    in_correct = False        # currently processed chunks is correct until now
    last_correct = 'O'        # previous chunk tag in corpus
    last_correct_type = ''    # type of previously identified chunk tag
    last_guessed = 'O'        # previously identified chunk tag
    last_guessed_type = ''    # type of previous chunk tag in corpus

    for line in iterable:
        line = line.rstrip('\r\n')
        if len(line) < 3:
            continue

        if options['delimiter'] == ANY_SPACE:
            features = line.split()
        else:
            features = line.split(options['delimiter'])

        if num_features is None:
            num_features = len(features)
        elif num_features != len(features) and len(features) != 0:
            raise FormatError('unexpected number of features: %d (%d)' %
                              (len(features), num_features))

        if len(features) == 0 or features[0] == options['boundary']:
            features = [options['boundary'], 'O', 'O']
        if len(features) < 3:
            raise FormatError('unexpected number of features in line %s' % line)

        guessed, guessed_type = parse_tag(features.pop())
        correct, correct_type = parse_tag(features.pop())
        first_item = features.pop(0)

        if first_item == options['boundary']:
            guessed = 'O'

        end_correct = end_of_chunk(last_correct, correct,
                                   last_correct_type, correct_type)
        end_guessed = end_of_chunk(last_guessed, guessed,
                                   last_guessed_type, guessed_type)
        start_correct = start_of_chunk(last_correct, correct,
                                       last_correct_type, correct_type)
        start_guessed = start_of_chunk(last_guessed, guessed,
                                       last_guessed_type, guessed_type)

        if in_correct:
            if end_correct and end_guessed and last_guessed_type == last_correct_type:
                in_correct = False
                counts.correct_chunk += 1
                counts.t_correct_chunk[last_correct_type] += 1
            elif end_correct != end_guessed or guessed_type != correct_type:
                in_correct = False

        if start_correct and start_guessed and guessed_type == correct_type:
            in_correct = True

        if start_correct:
            counts.found_correct += 1
            counts.t_found_correct[correct_type] += 1
        if start_guessed:
            counts.found_guessed += 1
            counts.t_found_guessed[guessed_type] += 1
        if first_item != options['boundary']:
            if correct == guessed and guessed_type == correct_type:
                counts.correct_tags += 1
            counts.token_counter += 1

        last_guessed = guessed
        last_correct = correct
        last_guessed_type = guessed_type
        last_correct_type = correct_type

    if in_correct:
        counts.correct_chunk += 1
        counts.t_correct_chunk[last_correct_type] += 1

    return counts


def uniq(iterable):
    seen = set()
    return [i for i in iterable if not (i in seen or seen.add(i))]


def calculate_metrics(correct, guessed, total):
    tp, fp, fn = correct, guessed-correct, total-correct
    p = 0 if tp + fp == 0 else 1.*tp / (tp + fp)
    r = 0 if tp + fn == 0 else 1.*tp / (tp + fn)
    f = 0 if p + r == 0 else 2 * p * r / (p + r)
    return Metrics(tp, fp, fn, p, r, f)


def metrics(counts):
    c = counts
    overall = calculate_metrics(
        c.correct_chunk, c.found_guessed, c.found_correct
    )
    by_type = {}
    for t in uniq(list(c.t_found_correct.keys()) + list(c.t_found_guessed.keys())):
        by_type[t] = calculate_metrics(
            c.t_correct_chunk[t], c.t_found_guessed[t], c.t_found_correct[t]
        )
    return overall, by_type


def get_scores(counts):
    """
    returns the scores for further processing
    :param counts: as generated by evaluate() above
    :return: map with type:[P, R, F1]
    """
    scores = {}
    overall, by_type = metrics(counts)
    for i, m in sorted(by_type.items()):
        scores[i] = [100.*m.prec, 100.*m.rec, 100.*m.fscore]
    return scores


def report(counts, out=None):
    """
    :param counts: as generated by evaluate() above!
    :param out: prints the scores
    :return:
    """
    out = ''

    overall, by_type = metrics(counts)

    c = counts
    out += ('processed %d tokens with %d phrases; ' %
              (c.token_counter, c.found_correct))
    out += ('found: %d phrases; correct: %d.\n' %
              (c.found_guessed, c.correct_chunk))

    if c.token_counter > 0:
        out += ('accuracy: %6.2f%%; ' %
                  (100.*c.correct_tags/c.token_counter))
        out += ('precision: %6.2f%%; ' % (100.*overall.prec))
        out += ('recall: %6.2f%%; ' % (100.*overall.rec))
        out += ('FB1: %6.2f\n' % (100.*overall.fscore))

    for i, m in sorted(by_type.items()):
        out += ('%17s: ' % i)
        out += ('precision: %6.2f%%; ' % (100.*m.prec))
        out += ('recall: %6.2f%%; ' % (100.*m.rec))
        out += ('FB1: %6.2f  %d\n' % (100.*m.fscore, c.t_found_guessed[i]))
    return out


def end_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk ended between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    # these chunks are assumed to have length 1
    if prev_tag == ']': chunk_end = True
    if prev_tag == '[': chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk started between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    # these chunks are assumed to have length 1
    if tag == '[': chunk_start = True
    if tag == ']': chunk_start = True

    return chunk_start
