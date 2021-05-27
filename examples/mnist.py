

def test_mnist():
    from sklearn.datasets import load_digits
    mnist = load_digits()

    tree = JPT(...)
    tree.fit(np.ascontiguousarray(mnist.data, dtype=np.float32),
             # np.asarray(mnist.target.reshape(-1, 1), dtype=np.float64))
             np.ascontiguousarray(mnist.data, dtype=np.float64))
    leaves = list(tree.leaves.values())

    rows = len(leaves) // 10 + 1
    cols = max(2, len(leaves))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 6))
    out(axes)
    if len(axes.shape) == 1:
        axes = np.array([axes])
    out(axes)
    # for i in range(20):
    # axes[i // 10, i % 10].imshow(mnist.images[i], cmap='gray')
    # axes[i // 10, i % 10].axis('off')
    # axes[i // 10, i % 10].set_title(f"target: {mnist.target[i]}")
    for i, leaf in enumerate(leaves):
        out(leaf.dist)
        model = leaf.dist.mean.reshape(8, 8)
        idx = i // 10, i % 10 if len(axes.shape) == 1 else i
        axes[idx].imshow(model, cmap='gray')
    plt.tight_layout()
    plt.show()