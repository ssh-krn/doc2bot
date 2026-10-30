"""Writes eval/qa.jsonl: 50 questions on the machine learning corpus.

Each question has gold passages (doc, page, snippet). Snippets are copied from the corpus source text, and
eval/validate_qa.py checks each one against the rendered PDF, so a wrong label fails the build.

Types:
  fact, definition, comparison, numeric  - answerable from one passage
  multi_hop                               - needs two passages from different documents
  unanswerable                            - the corpus does not contain the answer; a good system refuses

Run from backend/: python -m eval.build_qa
"""

import json
from pathlib import Path

I = "01_intro_to_ml.pdf"
R = "02_regression.pdf"
T = "03_trees_and_ensembles.pdf"
N = "04_neural_networks.pdf"
C = "05_cnn_rnn_transformers.pdf"
P = "06_evaluation_and_practice.pdf"


def g(doc, page, snippet):
    return {"doc": doc, "page": page, "snippet": snippet}


ITEMS = [
    # ---- definitions and facts, single passage ----
    ("definition", "What are the three main paradigms of machine learning?",
     [g(I, 1, "The three main paradigms are supervised learning, unsupervised learning, and reinforcement learning.")],
     ["supervised", "unsupervised", "reinforcement"]),
    ("comparison", "How does classification differ from regression in supervised learning?",
     [g(I, 1, "Classification predicts a discrete category, such as whether an email is spam, while regression predicts a continuous value, such as a house price.")],
     ["discrete", "continuous"]),
    ("fact", "What does unsupervised learning look for in unlabelled data?",
     [g(I, 1, "the algorithm looks for structure, for example by grouping customers into clusters")],
     ["structure"]),
    ("fact", "What does a reinforcement learning agent learn?",
     [g(I, 1, "it learns a policy that maximizes the total reward over time")],
     ["policy"]),
    ("definition", "What is overfitting?",
     [g(I, 2, "A model with high variance is very sensitive to the particular training sample and overfits, meaning it memorizes noise.")],
     ["noise"]),
    ("definition", "What is underfitting?",
     [g(I, 2, "A model with high bias makes strong assumptions and underfits, meaning it fails to capture the true pattern even in the training data.")],
     ["pattern"]),
    ("fact", "How is overfitting usually diagnosed?",
     [g(I, 2, "Overfitting is usually diagnosed when training error keeps falling while validation error starts to rise.")],
     ["validation"]),
    ("fact", "How is each example used in k-fold cross-validation?",
     [g(I, 2, "every example is used for validation exactly once")],
     ["once"]),
    ("definition", "How does linear regression make a prediction?",
     [g(R, 1, "Linear regression models the target as a weighted sum of input features plus a bias term.")],
     ["weighted sum"]),
    ("definition", "Which loss are linear regression weights usually chosen to minimize?",
     [g(R, 1, "The weights are usually chosen to minimize the mean squared error between predictions and true values.")],
     ["mean squared error"]),
    ("fact", "What does the learning rate control in gradient descent?",
     [g(R, 1, "The learning rate controls the size of each step.")],
     ["step"]),
    ("comparison", "What goes wrong when the learning rate is too large or too small?",
     [g(R, 1, "A learning rate that is too large can cause the loss to diverge, while one that is too small makes training slow.")],
     ["diverg", "slow"]),
    ("fact", "Why does feature scaling help gradient descent?",
     [g(R, 1, "often makes gradient descent converge much faster")],
     ["converge"]),
    ("definition", "Which function does logistic regression use to turn a score into a probability?",
     [g(R, 2, "It computes a linear score and passes it through the sigmoid function")],
     ["sigmoid"]),
    ("fact", "What loss does logistic regression minimize?",
     [g(R, 2, "which is equivalent to minimizing the binary cross-entropy loss")],
     ["cross-entropy"]),
    ("comparison", "Why is cross-entropy preferred over squared error for classification?",
     [g(R, 2, "Cross-entropy penalizes confident wrong predictions heavily, so it is preferred over squared error for classification.")],
     ["confident"]),
    ("fact", "What does softmax do when there are more than two classes?",
     [g(R, 2, "converts a vector of scores into a probability distribution that sums to one")],
     ["probability"]),
    ("comparison", "How do L1 and L2 regularization differ?",
     [g(R, 2, "L1 regularization penalizes their absolute values and can drive some weights to exactly zero")],
     ["absolute", "zero"]),
    ("definition", "What is information gain in a decision tree?",
     [g(T, 1, "information gain is the drop in entropy caused by a split")],
     ["entropy"]),
    ("comparison", "How does the Gini impurity compare with entropy?",
     [g(T, 1, "The Gini impurity is a similar measure that is faster to compute")],
     ["faster"]),
    ("fact", "Why do fully grown decision trees need care?",
     [g(T, 1, "a fully grown tree tends to overfit")],
     ["overfit"]),
    ("fact", "What does pruning do to a decision tree?",
     [g(T, 1, "Pruning limits the depth of the tree")],
     ["depth"]),
    ("definition", "What is a bootstrap sample?",
     [g(T, 2, "a random sample drawn with replacement from the training set")],
     ["replacement"]),
    ("comparison", "How do random forests differ from plain bagging?",
     [g(T, 2, "restricting each split to a random subset of features, which makes the individual trees less correlated")],
     ["features"]),
    ("comparison", "How does boosting train its models compared with bagging?",
     [g(T, 2, "models are trained one after another")],
     ["after another"]),
    ("fact", "What does gradient boosting fit each new tree to?",
     [g(T, 2, "fits each new tree to the negative gradient of the loss")],
     ["negative gradient"]),
    ("definition", "What is a perceptron?",
     [g(N, 1, "A single neuron with a step activation is the perceptron, which can only represent linearly separable functions.")],
     ["linearly separable"]),
    ("fact", "Why are hidden layers needed?",
     [g(N, 1, "adding at least one hidden layer lets the network represent non-linear functions")],
     ["non-linear"]),
    ("definition", "What does ReLU output for negative inputs?",
     [g(N, 1, "outputs zero for negative inputs and the input itself otherwise")],
     ["zero"]),
    ("comparison", "How does ReLU help compared with sigmoid in deep networks?",
     [g(N, 1, "reduces the vanishing gradient problem that affects sigmoid and tanh in deep networks")],
     ["vanishing"]),
    ("fact", "What rule does backpropagation apply layer by layer?",
     [g(N, 2, "It applies the chain rule layer by layer")],
     ["chain rule"]),
    ("definition", "What is momentum in stochastic gradient descent?",
     [g(N, 2, "Momentum accumulates a running average of past gradients to smooth the updates")],
     ["running average"]),
    ("definition", "What is one epoch of training?",
     [g(N, 2, "An epoch is one full pass over the training set")],
     ["full pass"]),
    ("numeric", "How many gates does a GRU have?",
     [g(C, 2, "The gated recurrent unit, or GRU, is a simpler design with two gates.")],
     ["two"]),
    ("definition", "What does a convolutional layer slide across an image?",
     [g(C, 1, "slides a small learned filter across an image")],
     ["filter"]),
    ("fact", "What does max pooling keep?",
     [g(C, 1, "Max pooling keeps the largest value in each window.")],
     ["largest"]),
    # ---- multi-hop: two passages from different documents ----
    ("multi_hop", "Which regularization technique can drive weights to exactly zero, and what does early stopping monitor?",
     [g(R, 2, "can drive some weights to exactly zero"),
      g(P, 2, "Early stopping halts training when validation loss stops improving.")],
     ["l1", "validation"]),
    ("multi_hop", "What does a high-variance model do, and which penalty on logistic regression weights counters that?",
     [g(I, 2, "A model with high variance is very sensitive to the particular training sample and overfits, meaning it memorizes noise."),
      g(R, 2, "L2 regularization, also called ridge, penalizes the sum of squared weights")],
     ["overfit", "regulariz"]),
    ("multi_hop", "Why is ReLU favoured over sigmoid, and what does batch normalization stabilize?",
     [g(N, 1, "reduces the vanishing gradient problem that affects sigmoid and tanh in deep networks"),
      g(P, 2, "which stabilizes and speeds up training")],
     ["vanishing", "stabiliz"]),
    ("multi_hop", "Name a deep-network remedy for overfitting and one general remedy that works for any model.",
     [g(P, 2, "Dropout randomly sets a fraction of activations to zero during training"),
      g(I, 2, "Common remedies include collecting more data, simplifying the model, adding regularization, and stopping training early.")],
     ["dropout", "more data"]),
    ("multi_hop", "Which activation feeds a sigmoid output for binary classification, and which metric summarizes ranking quality across thresholds?",
     [g(R, 2, "It computes a linear score and passes it through the sigmoid function"),
      g(P, 1, "the area under it, ROC AUC, summarizes ranking quality across all thresholds")],
     ["sigmoid", "roc auc"]),
    ("multi_hop", "Which direction does gradient descent move the weights, and what do gradient boosting trees fit?",
     [g(R, 1, "moves the weights in the direction opposite to the gradient of the loss"),
      g(T, 2, "fits each new tree to the negative gradient of the loss, the residual errors of the current ensemble")],
     ["opposite", "residual"]),
    # ---- unanswerable: the corpus does not cover these ----
    ("unanswerable", "Who won the 2018 FIFA World Cup?", [], []),
    ("unanswerable", "What is the boiling point of mercury?", [], []),
    ("unanswerable", "How do I file a tax return in India?", [], []),
    ("unanswerable", "What is the capital of Australia?", [], []),
    ("unanswerable", "Which programming language is the Linux kernel written in?", [], []),
    ("unanswerable", "What is the recommended adult dose of ibuprofen?", [], []),
    ("unanswerable", "How much did a Tesla Model 3 cost in 2024?", [], []),
    ("unanswerable", "What year was the Eiffel Tower completed?", [], []),
]


def main() -> None:
    out = Path("eval/qa.jsonl")
    lines = []
    for i, (qtype, question, golds, keywords) in enumerate(ITEMS, start=1):
        answerable = qtype != "unanswerable"
        lines.append(json.dumps({
            "id": f"q{i:03d}",
            "type": qtype,
            "question": question,
            "answerable": answerable,
            "golds": golds,
            "gold_answer_keywords": keywords,
        }, ensure_ascii=False))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} questions to {out}")


if __name__ == "__main__":
    main()
