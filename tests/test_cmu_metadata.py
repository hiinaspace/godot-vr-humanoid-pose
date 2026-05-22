from __future__ import annotations

from pathlib import Path

from validation.vr_pose_validation.cmu_metadata import (
    clip_metadata_from_subject_page,
    label_activity,
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

SPASTIC_SUBJECT_HTML = """
<table>
<tr><td>Subject #104 (motion) file index</td></tr>
<tr><td></td><td>18</td><td>SpasticStop</td></tr>
</table>
"""

MOTORCYCLE_SUBJECT_HTML = """
<table>
<tr><td>Subject #133 (Baby Styled Walk) file index</td></tr>
<tr><td></td><td>25</td><td>Motorcycle</td></tr>
</table>
"""


def test_label_environment_marks_motion_description_nonflat() -> None:
    split, tags, source = label_environment("Walking up and down stairs")

    assert split == "nonflat"
    assert tags == ("stairs",)
    assert source == "motion_description"


def test_label_activity_marks_common_vr_stationary_interaction() -> None:
    activity_class, tags, source = label_activity("punch/strike")

    assert activity_class == "dance_exercise"
    assert tags == ("martial_arts",)
    assert source == "motion_description"


def test_label_activity_marks_sitting_before_locomotion() -> None:
    activity_class, tags, source = label_activity("walk to chair, sit down")

    assert activity_class == "sitting"
    assert tags == ("chair", "sit")
    assert source == "motion_description"


def test_label_activity_marks_swimming_as_out_of_distribution_class() -> None:
    activity_class, tags, source = label_activity("Breast Stroke", "Swimming")

    assert activity_class == "swimming"
    assert tags == ("swimming",)
    assert source == "motion_description"


def test_label_environment_marks_swimming_as_nonflat_out_of_distribution() -> None:
    split, tags, source = label_environment("Free Style", "Swimming")

    assert split == "nonflat"
    assert tags == ("swimming",)
    assert source == "motion_description"


def test_swimming_context_does_not_relabel_unrelated_free_style_motion() -> None:
    split, _environment_tags, _environment_source = label_environment(
        "basketball - low, fast free style dribble, dribble through legs",
        "dribble, shoot basketball",
    )
    activity_class, tags, source = label_activity(
        "basketball - low, fast free style dribble, dribble through legs",
        "dribble, shoot basketball",
    )

    assert split == "flat"
    assert activity_class == "sports"
    assert tags == ("ball_sport",)
    assert source == "motion_description"


def test_label_activity_handles_cmu_camel_case_and_digit_suffixes() -> None:
    assert label_activity("RegularWalk8")[0] == "locomotion"
    assert label_activity("GetUpFaceDown")[0] == "floor_pose"


def test_parse_subject_page_labels_flat_and_nonflat_trials() -> None:
    rows = clip_metadata_from_subject_page("114", SUBJECT_HTML)

    by_clip = {row.clip: row for row in rows}
    assert by_clip["114_07"].environment_split == "nonflat"
    assert by_clip["114_07"].environment_tags == ("stairs",)
    assert by_clip["114_07"].activity_class == "locomotion"
    assert by_clip["114_13"].environment_split == "flat"
    assert by_clip["114_13"].activity_class == "locomotion"


def test_parse_subject_page_applies_spastic_stop_ood_clip_override() -> None:
    rows = clip_metadata_from_subject_page("104", SPASTIC_SUBJECT_HTML)

    row = rows[0]
    assert row.clip == "104_18"
    assert row.environment_split == "nonflat"
    assert row.environment_tags == ("suspended",)
    assert row.label_source == "manual_clip_override"
    assert row.activity_class == "locomotion"


def test_parse_subject_page_applies_motorcycle_data_issue_clip_override() -> None:
    rows = clip_metadata_from_subject_page("133", MOTORCYCLE_SUBJECT_HTML)

    row = rows[0]
    assert row.clip == "133_25"
    assert row.environment_split == "unknown"
    assert row.environment_tags == ("data_issue", "sideways_t_pose")
    assert row.activity_class == "unknown"
    assert row.activity_tags == ("parse_issue", "sideways_t_pose")
    assert row.activity_label_source == "manual_clip_override"


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
    assert metadata["01_01"].activity_class == "locomotion"
    assert metadata["126_04"].environment_split == "nonflat"
    assert metadata["126_04"].environment_tags == ("swimming",)
    assert metadata["126_04"].activity_class == "swimming"
    assert metadata["104_18"].environment_split == "nonflat"
    assert metadata["104_18"].environment_tags == ("suspended",)
    assert metadata["133_25"].environment_split == "unknown"
    assert metadata["133_25"].environment_tags == ("data_issue", "sideways_t_pose")
    assert metadata["133_25"].activity_class == "unknown"
