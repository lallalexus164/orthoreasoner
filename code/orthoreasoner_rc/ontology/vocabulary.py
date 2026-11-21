from dataclasses import dataclass


@dataclass(frozen=True)
class OntologyTerm:
    key: str
    anatomy: str
    phase: str
    parent: str
    polarity: float


TERMS: tuple[OntologyTerm, ...] = (
    OntologyTerm(
        "mri_edema_supraspinatus_active", "supraspinatus", "active_inflammation", "mri_edema", 1.0
    ),
    OntologyTerm(
        "mri_effusion_subacromial_active",
        "subacromial_bursa",
        "active_inflammation",
        "mri_effusion",
        0.8,
    ),
    OntologyTerm(
        "pdus_hyperemia_bursal_active", "bursa", "active_inflammation", "pdus_hyperemia", 1.0
    ),
    OntologyTerm("tear_depth_partial", "rotator_cuff", "structural_damage", "tear_depth", 0.4),
    OntologyTerm("tear_depth_full", "rotator_cuff", "structural_damage", "tear_depth", 1.0),
    OntologyTerm("goutallier_equivocal", "muscle", "fatty_infiltration", "goutallier", -0.5),
    OntologyTerm("goutallier_advanced", "muscle", "fatty_infiltration", "goutallier", -1.0),
    OntologyTerm("acute_trauma", "history", "acuity", "acuity", 1.0),
    OntologyTerm("chronic_degeneration", "history", "acuity", "acuity", -0.7),
    OntologyTerm("dominant_arm_high_demand", "function", "demand", "activity", 0.8),
    OntologyTerm("bursal_side_partial", "tendon_surface", "lesion_side", "bursal_side", 0.5),
    OntologyTerm("retracted_tendon", "tendon", "retraction", "retraction", 0.7),
)


def term_keys() -> tuple[str, ...]:
    return tuple(term.key for term in TERMS)


def term_by_key(key: str) -> OntologyTerm:
    for term in TERMS:
        if term.key == key:
            return term
    raise KeyError(key)
