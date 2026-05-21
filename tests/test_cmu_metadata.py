from __future__ import annotations

from pathlib import Path

from validation.vr_pose_validation.cmu_metadata import (
    clip_metadata_from_subject_page,
    label_environment,
    read_cmu_metadata,
)

SUBJECT_HTML = """
<table>
<tr><td>Subject #114 (Pregnant Woman) file index</td></tr>
<tr><td>Image</td><td>Trial #</td><td>Motion Description</td></tr>
<tr><td></td><td>7</td><td>Walking up and down stairs</td></tr>
<tr><td></td><td>13</td><td>Walk</td></tr>
</table>
"""

GENERIC_SUBJECT_HTML = """
<table>
<tr><td>Subject #122 (Varying Height and Length Steps) file index</td></tr>
<tr><td></td><td>1</td><td>clean</td></tr>
</table>
"""


def test_label_environment_marks_motion_description_nonflat() -> None:
    split, tags, source = label_environment("Walking up and down stairs")

    assert split == "nonflat"
    assert tags == ("stairs",)
    assert source == "motion_description"


def test_parse_subject_page_labels_flat_and_nonflat_trials() -> None:
    rows = clip_metadata_from_subject_page("114", SUBJECT_HTML)

    by_clip = {row.clip: row for row in rows}
    assert by_clip["114_07"].environment_split == "nonflat"
    assert by_clip["114_07"].environment_tags == ("stairs",)
    assert by_clip["114_13"].environment_split == "flat"


def test_generic_motion_description_can_inherit_subject_label() -> None:
    rows = clip_metadata_from_subject_page("122", GENERIC_SUBJECT_HTML)

    assert rows[0].environment_split == "nonflat"
    assert rows[0].environment_tags == ("step_obstacle",)
    assert rows[0].label_source == "subject_description"


def test_committed_cmu_metadata_file_is_readable() -> None:
    metadata = read_cmu_metadata(Path("validation/cmu_clip_metadata.json"))

    assert len(metadata) == 2514
    assert metadata["01_01"].environment_split == "nonflat"
    assert metadata["01_01"].environment_tags == ("playground",)
