import pytest

from drawing_classifier.data import make_loader, split_train_validation


def indices(subset):
    return list(subset.indices)


def test_split_by_count_and_fraction(small_dataset):
    train, validation = split_train_validation(small_dataset, 50, seed=1)
    assert (len(train), len(validation)) == (150, 50)
    train, validation = split_train_validation(small_dataset, 0.1, seed=1)
    assert (len(train), len(validation)) == (180, 20)


def test_split_is_disjoint_and_reproducible(small_dataset):
    train_a, validation_a = split_train_validation(small_dataset, 0.2, seed=42)
    train_b, validation_b = split_train_validation(small_dataset, 0.2, seed=42)
    assert indices(train_a) == indices(train_b)
    assert indices(validation_a) == indices(validation_b)
    assert not set(indices(train_a)) & set(indices(validation_a))
    _, validation_c = split_train_validation(small_dataset, 0.2, seed=7)
    assert indices(validation_c) != indices(validation_a)


@pytest.mark.parametrize("size", [0, 200, 0.0, 1.0, 1.5, -3])
def test_split_rejects_invalid_sizes(small_dataset, size):
    with pytest.raises(ValueError):
        split_train_validation(small_dataset, size, seed=0)


def test_shuffled_loader_order_follows_seed(small_dataset):
    first = [labels.tolist() for _, labels in make_loader(small_dataset, 32, shuffle=True, seed=3)]
    second = [labels.tolist() for _, labels in make_loader(small_dataset, 32, shuffle=True, seed=3)]
    assert first == second
