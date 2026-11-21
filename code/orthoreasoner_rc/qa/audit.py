from orthoreasoner_rc.experiments.paper_targets import TARGETS, tolerance
from orthoreasoner_rc.standards.aaos_rulebook import RULES, best_rule, decision_rules, family_rules


def validate_rule_keys() -> bool:
    keys = [rule.key for rule in RULES]
    return len(keys) == len(set(keys))


def count_rule_families() -> dict[str, int]:
    families = sorted({rule.family for rule in RULES})
    return {family: len(family_rules(family)) for family in families}


def count_decision_rules() -> dict[str, int]:
    return {decision: len(decision_rules(decision)) for decision in ("surgery", "conservative")}


def target_windows() -> dict[str, float]:
    return {f"{target.table}:{target.row}:{target.metric}": tolerance(target) for target in TARGETS}


def audit_reference_case() -> tuple[str, float]:
    rule, score = best_rule(0.8, 0.7, 0.2, 0.9, 0.8)
    return rule.key, score
