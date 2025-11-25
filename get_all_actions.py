from cyberwheel.red_actions.art_techniques import technique_mapping

def get_techniques():
    techniques = {}
    for _, cls in technique_mapping.items():
        if hasattr(cls, 'kill_chain_phases'):
            techniques[cls.__name__] = cls.kill_chain_phases

    return techniques

if __name__ == "__main__":
    file = "cyberwheel/data/configs/red_agent/rl_red_abstract.yaml"
    get_techniques()
    with open(file, "r") as f:
        data = f.read()
    techniques = get_techniques()

    new_data = ""
    for tech, phases in list(techniques.items()):
        if tech not in data:
            if 'discovery' in phases:
                phase = 'discovery'
                techniques[tech] = [phase]
                immediate_reward = 20
            elif 'impact' in phases:
                phase = 'impact'
                techniques[tech] = [phase]
                immediate_reward = 100
            elif 'privilege-escalation' in phases:
                phase = 'privilege-escalation'
                techniques[tech] = [phase]
                immediate_reward = 0
            elif 'lateral-movement' in phases:
                phase = 'lateral-movement'
                techniques[tech] = [phase]
                immediate_reward = 0
            else:
                del techniques[tech] 
                continue
            new_data += f"\n  {tech}:\n    class: {tech}\n    reward:\n      immediate: {immediate_reward}\n      recurring: 0\n    phase: {phase}\n"
    
    
    pingsweep_portscan = """\n  ARTPingSweep:\n    class: ARTPingSweep\n    reward:\n      immediate: 0\n      recurring: 0\n    phase: pingsweep\n  \n  ARTPortScan:\n    class: ARTPortScan\n    reward:\n      immediate: 0\n      recurring: 0\n    phase: portscan"""
    if "actions" not in data:
        combined_data = data + "\nactions:\n" + new_data
    else:   
        combined_data = data + new_data

    if "ARTPingSweep" not in data and "ARTPortScan" not in data:
        combined_data = combined_data + pingsweep_portscan

    with open(file, "w") as f:
        f.write(combined_data)

    dict_techniques = {}
    for tech, phases in techniques.items():
        dict_techniques.setdefault(f"{phases[0]}", []).append(tech)
    print("Mapping:\n",dict_techniques)