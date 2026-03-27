#!/usr/bin/env python3
"""Compute valid ART techniques by phase from exploitable service CVEs.

The script mirrors Cyberwheel's runtime logic:
1) Technique must be valid for the kill-chain phase/OS.
2) Technique must have at least one CVE overlap with the host/service CVEs.

run: python3 get_valid_techniques_from_services.py --format text

Discovery: 5 techniques
Lateral-movement: 8 techniques
Privilege-escalation: 5 techniques
Impact: 1 technique
"""
from __future__ import annotations

import sys
from pathlib import Path

import argparse
import json
import yaml

try:
    from cyberwheel.red_actions import art_techniques
    from cyberwheel.red_actions.actions.art_killchain_phase import ARTKillChainPhase
except ModuleNotFoundError:
    # Support running this file directly via:
    # python3 cyberwheel/utils/get_valid_techniques_from_services.py
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from cyberwheel.red_actions import art_techniques
    from cyberwheel.red_actions.actions.art_killchain_phase import ARTKillChainPhase


PHASE_ORDER = [
    "discovery",
    "lateral-movement",
    "privilege-escalation",
    "impact",
]


def _service_to_phase(service_name: str) -> str | None:
    lowered = service_name.lower()

    if "discovery" in lowered:
        return "discovery"
    if "impact" in lowered:
        return "impact"
    if any(token in lowered for token in ["lateralmovement", "lateral_movement", "lateral-movement"]):
        return "lateral-movement"
    if any(token in lowered for token in ["privilegeescalation", "privilege_escalation", "privilege-escalation"]):
        return "privilege-escalation"
    return None


def build_result(services: dict, os_name: str) -> dict:
    phase_service_cves = {phase: set() for phase in PHASE_ORDER}

    for service_name, service_data in services.items():
        phase = _service_to_phase(service_name)
        if phase is None:
            continue
        phase_service_cves[phase].update(service_data.get("cve", []))

    validity = ARTKillChainPhase.validity_mapping[os_name]
    valid_techniques = {}

    for phase in PHASE_ORDER:
        phase_cves = phase_service_cves[phase]
        matches = []

        for mitre_id in validity[phase]:
            technique = art_techniques.technique_mapping.get(mitre_id)
            if technique is None:
                continue

            technique_cves = getattr(technique, "cve_list", set()) or set()
            overlap = sorted(phase_cves & technique_cves)
            if overlap:
                matches.append(
                    {
                        "mitre_id": mitre_id,
                        "name": technique.name,
                        "matching_cves": overlap,
                    }
                )

        matches.sort(key=lambda item: item["mitre_id"])
        valid_techniques[phase] = matches

    return {
        "service_phase_cves": {phase: sorted(cves) for phase, cves in phase_service_cves.items()},
        "valid_techniques": valid_techniques,
    }


def print_text(result: dict) -> None:
    print("Valid techniques by phase\n")
    for phase in PHASE_ORDER:
        techniques = result["valid_techniques"][phase]
        print(f"[{phase}] ({len(techniques)})")
        if not techniques:
            print("  - none")
            continue

        for item in techniques:
            cves = ", ".join(item["matching_cves"])
            print(f"  - {item['mitre_id']}: {item['name']}")
            print(f"    CVEs: {cves}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--services",
        type=Path,
        default=Path("cyberwheel/data/configs/services/windows_exploitable_services.yaml"),
        help="Path to exploitable services YAML.",
    )
    parser.add_argument(
        "--os",
        dest="os_name",
        default="windows",
        choices=sorted(ARTKillChainPhase.validity_mapping.keys()),
        help="OS validity mapping to use.",
    )
    parser.add_argument(
        "--format",
        default="text",
        choices=["text", "json"],
        help="Output format.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    with args.services.open("r", encoding="utf-8") as handle:
        services = yaml.safe_load(handle) or {}

    result = build_result(services, args.os_name)

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print_text(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
