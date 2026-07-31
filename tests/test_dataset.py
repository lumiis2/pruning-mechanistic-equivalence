import torch

from src.data.dataset import ShapeColorDataset, create_splits

DATA_CONFIG = {
    "image_size": 32,
    "train_size": 10_000,
    "val_size": 2_000,
    "test_size": 4_000,
    "train_alignment_probability": 0.95,
    "val_alignment_probability": 0.95,
    "min_shape_size": 10,
    "max_shape_size": 16,
}


def make_dataset(split, seed=0):
    return create_splits(seed=seed, data_config=DATA_CONFIG)[split]


def test_split_sizes_and_exact_group_counts():
    splits = create_splits(seed=42, data_config=DATA_CONFIG)
    assert {name: len(dataset) for name, dataset in splits.items()} == {
        "train_biased": 10_000,
        "val_biased": 2_000,
        "test_balanced": 4_000,
    }
    assert torch.bincount(splits["train_biased"].groups, minlength=4).tolist() == [4750, 250, 250, 4750]
    assert torch.bincount(splits["val_biased"].groups, minlength=4).tolist() == [950, 50, 50, 950]
    assert torch.bincount(splits["test_balanced"].groups, minlength=4).tolist() == [1000] * 4


def test_example_schema_and_conventions():
    dataset = make_dataset("test_balanced", seed=7)
    example = dataset[0]
    assert set(example) == {"image", "label", "shape", "color", "aligned", "group"}
    assert example["image"].shape == (3, 32, 32)
    assert example["image"].dtype == torch.float32
    assert example["label"] == example["shape"]
    assert example["aligned"] == (example["color"] == example["label"])
    assert example["group"] == 2 * example["label"] + example["color"]


def test_same_seed_produces_identical_data():
    first = make_dataset("train_biased", seed=123)
    second = make_dataset("train_biased", seed=123)
    for index in (0, 17, 9999):
        assert first[index].keys() == second[index].keys()
        assert torch.equal(first[index]["image"], second[index]["image"])
        assert {k: first[index][k] for k in first[index] if k != "image"} == {
            k: second[index][k] for k in second[index] if k != "image"
        }


def test_biased_splits_are_class_balanced_and_95_percent_aligned():
    for split in ("train_biased", "val_biased"):
        dataset = make_dataset(split, seed=0)
        labels = dataset.groups // 2
        colors = dataset.groups % 2
        assert torch.bincount(labels).tolist() == [len(dataset) // 2] * 2
        assert abs((labels == colors).float().mean().item() - 0.95) < 1e-6


def test_color_permutation_has_exactly_balanced_groups_in_training():
    config = {**DATA_CONFIG, "train_alignment_probability": 0.5, "val_alignment_probability": 0.5}
    dataset = create_splits(seed=42, data_config=config)["train_biased"]
    assert torch.bincount(dataset.groups, minlength=4).tolist() == [2500] * 4


def test_shape_only_removes_rgb_color_information():
    dataset = create_splits(seed=42, data_config=DATA_CONFIG, variant="shape_only")["test_balanced"]
    image = dataset[0]["image"]
    assert torch.equal(image[0], image[1])
    assert torch.equal(image[1], image[2])


def test_color_only_uses_the_same_visual_shape_for_both_labels():
    dataset = create_splits(seed=42, data_config=DATA_CONFIG, variant="color_only")["test_balanced"]
    for target_label in (0, 1):
        index = int(((dataset.groups // 2) == target_label).nonzero(as_tuple=False)[0])
        image = dataset[index]["image"]
        foreground = image.ne(0.5).any(dim=0)
        occupied_rows = foreground.any(dim=1).nonzero(as_tuple=False).flatten()
        occupied_columns = foreground.any(dim=0).nonzero(as_tuple=False).flatten()
        expected_area = len(occupied_rows) * len(occupied_columns)
        assert int(foreground.sum()) == expected_area
