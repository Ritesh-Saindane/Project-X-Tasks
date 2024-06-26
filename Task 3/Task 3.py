import math
def count_ngrams(words, n):
    ngrams = []
    for i in range(len(words) - n + 1):
        ngrams.append(tuple(words[i:i + n]))
    return ngrams

def calculate_precision(candidate, references, n):
    candidate_ngrams = count_ngrams(candidate, n)
    reference_ngrams = []
    for ref in references:
        reference_ngrams.extend(count_ngrams(ref, n))

    match_count = sum(1 for ngram in candidate_ngrams if ngram in reference_ngrams)
    return match_count / max(1, len(candidate_ngrams))

def calculate_brevity_penalty(candidate, references):
    c_len = len(candidate)
    r_len = min(len(ref) for ref in references)
    if c_len >= r_len:
        return 1
    else:
        return math.exp(1 - r_len / c_len)

def bleu_score(references, candidate):
    references = [ref.split() for ref in references]
    candidate = candidate.split()

    precisions = []
    for n in range(1, 5):
        precision = calculate_precision(candidate, references, n)
        if precision > 0:
            precisions.append(precision)

    if not precisions:
        return 0

    precision_product = sum(math.log(p) for p in precisions) / len(precisions)

    brevity_penalty = calculate_brevity_penalty(candidate, references)

    return brevity_penalty * math.exp(precision_product)

references1 = [
    "It is a guide to action that ensures that the military will forever heed Party commands",
    "It is the guiding principle which guarantees the military forces always being under the command of the Party",
    "It is the practical guide for the army always to heed the directions of the party"
]
candidate1 = "It is a guide to action which ensures that the military always obeys the commands of the party"

references2 = [
    "It is a guide to action that ensures that the military will forever heed Party commands",
    "It is the guiding principle which guarantees the military forces always being under the command of the Party",
    "It is the practical guide for the army always to heed the directions of the party"
]
candidate2 = "It is the to action the troops forever hearing the activity guidebook that party direct"

print("BLEU score for test case 1:", bleu_score(references1, candidate1))
print("BLEU score for test case 2:", bleu_score(references2, candidate2))
