from cyberwheel.red_actions.art_techniques import technique_mapping

def get_techniques():
    techniques = {}
    for _, cls in technique_mapping.items():
        if hasattr(cls, 'kill_chain_phases'):
            techniques[cls.__name__] = cls.kill_chain_phases
    return techniques

if __name__ == "__main__":
    file = "cyberwheel/data/configs/red_agent/rl_red_complex.yaml"
    with open(file, "r") as f:
        data = f.read()
    techniques = get_techniques()

    new_data = ""
    for tech, phases in list(techniques.items()):
        if tech == 'NetworkServiceDiscovery':
            phase = 'pingsweep'
            tech = 'ARTPingSweep'
            techniques[tech] = [phase]
            immediate_reward = 5
        elif tech == 'RemoteSystemDiscovery':
            phase = 'portscan'
            tech = 'ARTPortScan'
            techniques[tech] = [phase]
            immediate_reward = 5
        elif 'discovery' in phases:
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
            immediate_reward = 40
        elif 'lateral-movement' in phases:
            phase = 'lateral-movement'
            techniques[tech] = [phase]
            immediate_reward = 5
        else: # do not include techniques that do not fit into these phases
            del techniques[tech] 
            continue
        
        if tech not in data:
            new_data += f"\n  {tech}:\n    class: {tech}\n    reward:\n      immediate: {immediate_reward}\n      recurring: 0\n    phase: {phase}\n"
    
        
    if "actions" not in data:
        combined_data = data + "\nactions:\n" + new_data
    else:   
        combined_data = data + new_data

    with open(file, "w") as f:
        f.write(combined_data)

    tech_phases_dict = {}
    total = 0
    for tech, phases in techniques.items():
        if phases[0] not in tech_phases_dict:
            tech_phases_dict[phases[0]] = []
        tech_phases_dict[phases[0]].append(tech)
        total += 1
    for phase, tech in tech_phases_dict.items():
        print(f"Phase {phase} of size {len(tech)}")
    print(f"Total techniques included: {total}")