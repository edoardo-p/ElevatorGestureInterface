from itertools import product

from matplotlib import pyplot as plt


def plot_confusion_matrix(conf_mat, labels=None):
    size = len(conf_mat)
    plt.figure(figsize=(10, 10))

    if labels is None:
        labels = [str(i) for i in range(size)]
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i, j in product(range(size), repeat=2):
        plt.text(i, j, f"{conf_mat[j, i]}", ha="center", va="center")

    plt.imshow(conf_mat, aspect="auto")
    plt.show()
