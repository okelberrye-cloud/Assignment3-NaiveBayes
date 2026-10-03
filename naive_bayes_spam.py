# ============================================================
# Assignment 3: Naive Bayes Spam Classifier
# Ethan Okelberry
#
# Goal: look at a text message and decide if it is "spam"
# (junk / scam) or "ham" (a normal message from a real person).
#
# How to run:  python naive_bayes_spam.py
# (SpamDetection.csv needs to be in the same folder as this file)
#
# No machine learning libraries are used here. Everything is
# done by hand with plain Python (dictionaries, lists, loops).
# ============================================================

import csv      # built in Python module, just helps us read the .csv file
import string   # built in Python module, gives us a list of punctuation marks


# ------------------------------------------------------------
# STEP 1: Load the dataset and split it into training and testing
# ------------------------------------------------------------
# The csv file has two columns:
#   Target -> the label ("spam" or "ham")
#   data   -> the actual text message
#
# The assignment says the first 20 messages are the TRAINING set
# (the messages our model learns from) and the last 10 are the
# TESTING set (messages the model has never seen, so we can check
# if it actually learned anything).

def load_dataset(filename):
    messages = []  # this will hold pairs like ("spam", "WINNER you won...")
    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # skip the first row because it is just the column names
        for row in reader:
            label = row[0].strip()
            text = row[1].strip()
            messages.append((label, text))
    return messages


# ------------------------------------------------------------
# Helper: turn a sentence into a list of words
# ------------------------------------------------------------
# Computers see "Call" and "call" as two different words, so we
# make everything lowercase. We also strip punctuation so "free."
# and "free" count as the same word. Then we split on spaces.
#
# Example: "Call me now!" -> ["call", "me", "now"]
#
# This is called a "bag of words" because we only care WHICH words
# show up and HOW MANY times, not what order they are in.

def tokenize(sentence):
    sentence = sentence.lower()
    for mark in string.punctuation:
        sentence = sentence.replace(mark, "")
    return sentence.split()


# ------------------------------------------------------------
# STEP 2: Prior probabilities, P(spam) and P(ham)
# ------------------------------------------------------------
# The "prior" is what we believe BEFORE reading the message at all.
# It just asks: out of all the training messages, what fraction
# were spam and what fraction were ham?
#
#   P(spam) = number of spam messages / total messages
#   P(ham)  = number of ham messages  / total messages
#
# Think of it like a starting guess. If most of the training
# messages were spam, the model leans toward spam a little before
# it even looks at the words.

def compute_priors(training_set):
    total = len(training_set)
    spam_count = 0
    ham_count = 0
    for label, text in training_set:
        if label == "spam":
            spam_count += 1
        else:
            ham_count += 1
    p_spam = spam_count / total
    p_ham = ham_count / total
    return p_spam, p_ham, spam_count, ham_count


# ------------------------------------------------------------
# STEP 3: Conditional probabilities, P(word|spam) and P(word|ham)
# ------------------------------------------------------------
# P(word|spam) means "if we already know the message is spam, how
# likely is it to contain this word?"
#
# To get this we count how many times each word shows up in all
# the spam messages, and do the same for ham.
#
# Basic formula:
#   P(word|spam) = (times word appears in spam) / (total words in spam)
#
# PROBLEM: if a word never showed up in spam during training, that
# probability is 0. And since we multiply probabilities together
# later, one 0 would make the whole message's score 0, which is way
# too harsh for just one word.
#
# FIX = Laplace smoothing (also called "add one" smoothing).
# We pretend every word in the vocabulary showed up one extra time:
#
#   P(word|spam) = (count of word in spam + 1) / (total spam words + vocabulary size)
#
# The "+ vocabulary size" on the bottom balances out all the +1's
# we added on top, so the probabilities still add up to 1.
#
# The vocabulary is every unique word from the training set
# (spam and ham combined).

def compute_word_counts(training_set):
    spam_word_counts = {}   # dictionary: word -> how many times it shows up in spam
    ham_word_counts = {}    # dictionary: word -> how many times it shows up in ham
    vocabulary = set()      # a set only keeps unique words, no repeats

    for label, text in training_set:
        words = tokenize(text)
        for word in words:
            vocabulary.add(word)
            if label == "spam":
                spam_word_counts[word] = spam_word_counts.get(word, 0) + 1
            else:
                ham_word_counts[word] = ham_word_counts.get(word, 0) + 1

    total_spam_words = sum(spam_word_counts.values())
    total_ham_words = sum(ham_word_counts.values())
    return spam_word_counts, ham_word_counts, vocabulary, total_spam_words, total_ham_words


def word_given_class(word, word_counts, total_words_in_class, vocab_size):
    # Laplace smoothing formula from above.
    # .get(word, 0) means "give me the count, or 0 if we never saw it"
    count = word_counts.get(word, 0)
    return (count + 1) / (total_words_in_class + vocab_size)


# ------------------------------------------------------------
# STEP 4: P(sentence|class) and the posterior probabilities
# ------------------------------------------------------------
# Now for a test sentence we want P(sentence|spam), meaning how
# likely the whole sentence is if it were spam.
#
# This is where the "naive" part of Naive Bayes comes in. We assume
# every word is independent of the others, so we can just multiply
# each word's probability together:
#
#   P(sentence|spam) = P(word1|spam) * P(word2|spam) * P(word3|spam) * ...
#
# Then we use Bayes' rule to flip it around and get what we actually
# want, which is the chance the message IS spam given its words:
#
#   P(spam|sentence) is proportional to P(spam) * P(sentence|spam)
#   P(ham|sentence)  is proportional to P(ham)  * P(sentence|ham)
#
#   posterior = prior * conditional
#
# "Proportional to" means these two numbers are not real
# probabilities yet (they are super tiny and do not add to 1). To
# turn them into real percentages, we divide each one by their sum.
# That is called normalizing. Whichever one is bigger wins.
#
# Words in the test sentence that were never in training still get
# the Laplace formula with a count of 0, so they get a small
# probability instead of 0.

def classify(sentence, p_spam, p_ham, spam_counts, ham_counts,
             total_spam_words, total_ham_words, vocab_size):
    words = tokenize(sentence)

    # start at 1 because we are multiplying (starting at 0 would make everything 0)
    p_sentence_given_spam = 1.0
    p_sentence_given_ham = 1.0

    for word in words:
        p_sentence_given_spam *= word_given_class(word, spam_counts, total_spam_words, vocab_size)
        p_sentence_given_ham *= word_given_class(word, ham_counts, total_ham_words, vocab_size)

    # posterior is proportional to prior * conditional
    spam_score = p_spam * p_sentence_given_spam
    ham_score = p_ham * p_sentence_given_ham

    # normalize so the two posteriors add up to 1 (100%)
    total_score = spam_score + ham_score
    posterior_spam = spam_score / total_score
    posterior_ham = ham_score / total_score

    if posterior_spam > posterior_ham:
        predicted = "spam"
    else:
        predicted = "ham"

    return {
        "p_sentence_given_spam": p_sentence_given_spam,
        "p_sentence_given_ham": p_sentence_given_ham,
        "spam_score": spam_score,
        "ham_score": ham_score,
        "posterior_spam": posterior_spam,
        "posterior_ham": posterior_ham,
        "predicted": predicted,
    }


# ------------------------------------------------------------
# Main program: run every step in order
# ------------------------------------------------------------
def main():
    # STEP 1: load and split
    data = load_dataset("SpamDetection.csv")
    training_set = data[:20]   # first 20 messages
    testing_set = data[20:]    # last 10 messages
    print("STEP 1: Load and split")
    print("Total messages:", len(data))
    print("Training messages:", len(training_set))
    print("Testing messages:", len(testing_set))
    print()

    # STEP 2: priors
    p_spam, p_ham, spam_count, ham_count = compute_priors(training_set)
    print("STEP 2: Prior probabilities (from the training set)")
    print(f"P(spam) = {spam_count}/{len(training_set)} = {p_spam:.2f}")
    print(f"P(ham)  = {ham_count}/{len(training_set)} = {p_ham:.2f}")
    print()

    # STEP 3: conditional probabilities with Laplace smoothing
    spam_counts, ham_counts, vocabulary, total_spam_words, total_ham_words = compute_word_counts(training_set)
    vocab_size = len(vocabulary)
    print("STEP 3: Conditional probabilities with Laplace smoothing")
    print("Vocabulary size |V| =", vocab_size)
    print("Total words in spam messages =", total_spam_words)
    print("Total words in ham messages  =", total_ham_words)
    print(f"Formula: P(word|spam) = (count + 1) / ({total_spam_words} + {vocab_size})")
    print(f"Formula: P(word|ham)  = (count + 1) / ({total_ham_words} + {vocab_size})")
    print("A few example words:")
    for word in ["call", "free", "prize", "you", "i", "am"]:
        ps = word_given_class(word, spam_counts, total_spam_words, vocab_size)
        ph = word_given_class(word, ham_counts, total_ham_words, vocab_size)
        print(f"  {word:>6}:  P(word|spam) = {ps:.4f}   P(word|ham) = {ph:.4f}")
    print()

    # STEPS 4 and 5: classify each test sentence and show the results
    print("STEPS 4 and 5: Classify each sentence in the test set")
    print("=" * 70)
    correct = 0
    for i, (actual_label, sentence) in enumerate(testing_set, start=1):
        result = classify(sentence, p_spam, p_ham, spam_counts, ham_counts,
                          total_spam_words, total_ham_words, vocab_size)

        print(f"Test sentence {i}: {sentence}")
        print(f"  P(sentence|spam) = {result['p_sentence_given_spam']:.3e}")
        print(f"  P(sentence|ham)  = {result['p_sentence_given_ham']:.3e}")
        print(f"  P(spam|sentence) = {result['posterior_spam']:.4f}   (unnormalized: {result['spam_score']:.3e})")
        print(f"  P(ham|sentence)  = {result['posterior_ham']:.4f}   (unnormalized: {result['ham_score']:.3e})")
        print(f"  Predicted class: {result['predicted']}   |   Actual class: {actual_label}")
        print("-" * 70)

        if result["predicted"] == actual_label:
            correct += 1

    # STEP 6: accuracy
    # Accuracy = number of sentences predicted correctly / total sentences
    accuracy = correct / len(testing_set)
    print()
    print("STEP 6: Test set accuracy")
    print(f"Accuracy = {correct}/{len(testing_set)} = {accuracy:.2f} ({accuracy * 100:.0f}%)")


# This line just means "run main() when this file is run directly"
if __name__ == "__main__":
    main()


# ============================================================
# Some limitations I noticed with this approach
# ============================================================
# 1. The "naive" assumption is not really true. Words in a sentence
#    DO depend on each other ("call now" and "claim prize" go
#    together a lot in spam). Naive Bayes ignores that and treats
#    every word separately. It still works pretty well anyway.
# 2. 20 training messages is a really tiny dataset, so a lot of test
#    words were never seen before. Those words all get the same small
#    smoothed probability, so they don't really help either side.
# 3. Multiplying a lot of small numbers together gets REALLY small
#    fast. With these short messages it's fine, but with long emails
#    the number could get so small the computer rounds it to 0. The
#    usual fix is adding logarithms instead of multiplying.
# 4. Because spam and ham have a different total number of words,
#    the denominator in the smoothing formula is different for each
#    class. So an unseen word slightly favors whichever class had
#    fewer total words.


# ============================================================
# AI Use Disclosure
# ============================================================
# I used Claude (an AI assistant) to help me with this assignment. I
# came in understanding the big picture of Naive Bayes from lecture,
# and I used Claude to help write the Python code and walk me through
# each step so I understood why it works, not just that it runs.
#
# I went through it step by step. First I split the 30 messages into
# 20 for training and 10 for testing. Then I found the priors by
# counting how many training messages were spam (9) and ham (11).
# After that I worked through how P(word|spam) and P(word|ham) are
# counted, and why Laplace smoothing adds 1 to every count so a word
# we never saw doesn't zero out the whole sentence. Then I followed
# how the naive assumption lets us multiply the word probabilities
# together, and how multiplying that by the prior gives the
# posterior for each class.
#
# The part that clicked most for me was test sentence 3. It's spam,
# but the model called it ham. When I looked into why, it was because
# "I" and "am" show up a lot in the ham training messages, and seven
# of its words (like "awarded" and "bonus") never showed up in
# training at all. Since ham has fewer total words, those unseen
# words get a slightly higher smoothed probability under ham, which
# pushed the answer the wrong way. Seeing that showed me how the
# formula actually affects the final decision.
#
# Claude also helped with the wording of the comments and the report.
# I reviewed all of the code and the results, and I can explain each
# step of how the model decides what is spam and what isn't.
