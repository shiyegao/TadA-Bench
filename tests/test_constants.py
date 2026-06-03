from src.constants import OFFICIAL_HF_DATASET, OFFICIAL_HF_DATASET_REVISION


def test_official_dataset_constants():
    assert OFFICIAL_HF_DATASET == "JinGao/TadA-Bench"
    assert len(OFFICIAL_HF_DATASET_REVISION) == 40
    int(OFFICIAL_HF_DATASET_REVISION, 16)
