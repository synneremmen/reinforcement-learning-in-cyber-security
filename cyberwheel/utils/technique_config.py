"""
Utility module for loading and managing host-type-to-technique mappings.

This module provides functionality to load technique mappings that define which MITRE
attack techniques are valid/allowed for different server types (WEB_SERVER, MAIL_SERVER,
FILE_SERVER, PROXY_SERVER, SSH_JUMP_SERVER) across different kill chain phases
(discovery, lateral_movement, privilege_escalation, impact).

The mapping is configuration-driven via YAML and enables server-type-specific attack
constraints during red team simulation.
"""

import yaml
from importlib.resources import files
from typing import Dict, List, Optional


_host_type_technique_mapping = None


def load_host_type_technique_mapping() -> Dict[str, Dict[str, List[str]]]:
    """
    Load the host-type to technique mapping from YAML configuration.
    
    This is cached after first load to avoid repeated file I/O.
    
    Returns:
        Dict mapping structure:
        {
            "WEB_SERVER": {
                "discovery": ["T1087.002", "T1046", ...],
                "lateral_movement": ["T1021.003", ...],
                "privilege_escalation": [...],
                "impact": [...]
            },
            "MAIL_SERVER": {...},
            ...
        }
    """
    global _host_type_technique_mapping
    
    if _host_type_technique_mapping is not None:
        return _host_type_technique_mapping
    
    try:
        config_path = files("cyberwheel.data.configs").joinpath('host_type_technique_mapping.yaml')
        with open(config_path, 'r') as f:
            mapping = yaml.safe_load(f)
        
        _host_type_technique_mapping = mapping
        return mapping
    
    except FileNotFoundError as e:
        print(f"ERROR: Could not find host_type_technique_mapping.yaml: {e}")
        # Return empty mapping to prevent crashes; filtering will gracefully degrade
        return {}
    except yaml.YAMLError as e:
        print(f"ERROR: Failed to parse host_type_technique_mapping.yaml: {e}")
        return {}


def get_allowed_techniques(host_type: str, phase: str) -> List[str]:
    """
    Get the list of allowed MITRE technique IDs for a given host type and phase.
    
    Args:
        host_type: The host type as a string (e.g., "WEB_SERVER", "MAIL_SERVER")
        phase: The kill chain phase (e.g., "discovery", "lateral_movement", "privilege_escalation", "impact")
    
    Returns:
        List of MITRE technique IDs allowed for this host type + phase combination.
        Returns empty list if host type or phase not found in mapping.
    """
    mapping = load_host_type_technique_mapping()
    
    if not mapping:
        return []
    # print(f"Retrieving allowed techniques for host type '{host_type}' and phase '{phase}'")
    host_type_str = str(host_type)  # Handle HostType enum
    return mapping.get(host_type_str, {}).get(phase, [])


def is_technique_allowed(host_type: str, phase: str, mitre_id: str) -> bool:
    """
    Check if a specific MITRE technique is allowed for a given host type and phase.
    
    Args:
        host_type: The host type (e.g., "WEB_SERVER")
        phase: The kill chain phase
        mitre_id: The MITRE technique ID (e.g., "T1046")
    
    Returns:
        True if the technique is in the allowed list, False otherwise.
    """
    allowed = get_allowed_techniques(host_type, phase)
    return mitre_id in allowed
