import importlib
import yaml
import numpy as np

from typing import Iterable

from cyberwheel.network.network_base import Network, Host
from cyberwheel.observation import RedObservation
from cyberwheel.red_actions.actions import (
    ARTPingSweep,
    ARTPortScan,
    ARTDiscovery,
    ARTLateralMovement,
    ARTPrivilegeEscalation,
    ARTImpact,
    Nothing
)
from cyberwheel.red_actions.red_base import RedActionResults
from cyberwheel.red_agents import ARTAgent
from cyberwheel.red_agents.red_agent_base import RedAgentResult
from cyberwheel.reward.reward_base import RewardMap


class RLComplexAgent(ARTAgent):
    """
    Filler
    """

    def __init__(self, network: Network, args) -> None:
        super().__init__(network, args, service_mapping=args.service_mapping)
        self.observation = RedObservation(args, network)
        self.observation.add_host(self.current_host.name, on_host=True)
        self.current_step = 0
    
    def from_yaml(self) -> None:
        contents = self.args.agent_config["red"]

        # Get module import path
        self.killchain = [
            ARTPingSweep,
            ARTPortScan,
            ARTDiscovery,
            ARTLateralMovement,
            ARTPrivilegeEscalation,
            ARTImpact,
        ]

        self.reward_map = {}

        self.entry_host: Host = contents["entry_host"]
        self.current_host : Host = self.network.hosts[self.entry_host] if self.entry_host.lower() != "random" else self.network.get_random_user_host()

        # Initialize the action space
        as_class = contents['action_space']
        asm = importlib.import_module("cyberwheel.red_agents.action_space")
        # self.action_space = getattr(asm, as_class)(self.killchain, self.current_host.name)

        self.reward_map["nothing"] = (0, 0)

        self.actions = []
        self.phase_map = {}
        for k, v in contents['actions'].items():
            self.reward_map[k] = (v['reward']["immediate"], v['reward']["recurring"])
            self.phase_map[k] = v['phase']

            try:
                cls = getattr(importlib.import_module("cyberwheel.red_actions.art_techniques"), k) 
                cls.name = cls.__name__
                self.actions.append(cls)
            except AttributeError:
                cls = getattr(importlib.import_module("cyberwheel.red_actions.actions"), k)
                cls.name = cls.__name__
                self.actions.append(cls)
        
        # Reinitialize action space with specific actions
        with open("reward_map.txt", "w") as f:
            data = yaml.dump(self.reward_map)
            f.write(data)

        self.action_space = getattr(asm, as_class)(self.actions, self.current_host.name)
        # print(len(self.actions), "actions loaded into RLComplexAgent action space.")

        self.phase_classes = {
            'discovery': ARTDiscovery,
            'lateral-movement': ARTLateralMovement,
            'privilege-escalation': ARTPrivilegeEscalation,
            'impact': ARTImpact,
            'pingsweep': ARTPingSweep,
            'portscan': ARTPortScan,
        }

        self.leader: Host = contents.get("leader", "random")
        if self.leader.lower() == "random_user":
            self.leader_host = self.network.get_random_user_host()
        elif self.leader.lower() == "random_server":
            self.leader_host = self.network.get_random_server_host()
        elif self.leader.lower() == "random":
            self.leader_host = self.network.get_random_host()
        else:
            self.leader_host = self.network.hosts[self.leader]

    def act(self, action: int) -> RedAgentResult:
        art_action, target_host_name = self.action_space.select_action(
            action
        )  # Selects ART Action, should include the action and target host
        source_host = self.current_host
        target_host = self.network.hosts[target_host_name] if target_host_name != "nothing" else self.current_host
        success = False
        # print("Validating action:", art_action.__name__, "on target:", target_host_name)
        if self.validate_action(art_action, target_host_name):
            # print("\n\nAction {} valid.".format(art_action.__name__))
            if art_action == ARTPingSweep or art_action == ARTPortScan:
                result = art_action(
                    self.current_host, target_host
                ).sim_execute()  # Executes the ART Action, returns results
            elif art_action == Nothing:
                result = Nothing(self.current_host, self.current_host).sim_execute()
            else:
                # For specific techniques
                # print("Executing technique ", art_action.mitre_id)
                phase = self.phase_map[art_action.__name__]
                phase_class = self.phase_classes.get(phase)
                if phase_class:
                    result = phase_class(self.current_host, target_host, [art_action.mitre_id]).sim_execute()
                else:
                    print("Phase class not found for phase:", phase)
                    result = RedActionResults(source_host, target_host)
            success = result.attack_success
            # print(f"Success: {success}\n\n")
            self.handle_action(result)
        else:
            result = RedActionResults(source_host, target_host) # will this be false by default?
        self.current_step += 1
        return RedAgentResult(
            art_action, 
            source_host, 
            target_host, 
            success, 
            self.get_observation_space(),
            result
        )  # Returns what ARTAgent act() should, probably. Or the observation space?

    def handle_action(self, result: RedActionResults) -> None:
        self.observation.update_obs(current_step=self.current_step, total_steps=self.args.num_steps)
        if not result.attack_success:
            return
        action = result.action
        src_host = result.src_host.name
        target_host = result.target_host.name
        if isinstance(action, type):
            phase = self.phase_map.get(action.__name__, '')
        else:
            phase = ''
        if phase == "pingsweep":  # Adds pingsweeped hosts to obs
            self.observation.update_host(target_host, sweeped=True)
            hosts = result.metadata[result.target_host.subnet.name]["sweeped_hosts"]
            for h in hosts:
                h_name = h.name
                if h_name in self.observation.obs.keys():
                    continue
                sweeped = h.subnet.name == result.target_host.subnet.name
                self.observation.add_host(h_name, sweeped=sweeped)
                self.action_space.add_host(h_name)
        elif phase == "portscan":  # Scans target host
            self.observation.update_host(target_host, scanned=True)
        elif phase == "discovery":  # Discovers host type
            self.observation.update_host(target_host, discovered=True, type=result.target_host.host_type.type)
        elif phase == "lateral-movement":  # Moves to target host
            self.observation.update_host(target_host, on_host=True)
            self.observation.update_host(src_host, on_host=False)
            self.current_host = result.target_host
        elif phase == "privilege-escalation":
            self.observation.update_host(target_host, escalated=True)
        elif phase == "impact":
            self.observation.update_host(target_host, impacted=True)

    def handle_network_change(self):
        current_hosts = self.network.hosts.keys()
        new_hosts = current_hosts - self.tracked_hosts
        for h in new_hosts:
            host = self.network.hosts[h]
            self.service_mapping[h] = self.get_valid_techniques_by_host(
                host, self.all_kcps
            )
            self.observation.add_host(h, sweeped=True)
            self.action_space.add_host(h)
        self.tracked_hosts = current_hosts
    

    def validate_action(self, action, target_host: str) -> bool:
        if action == Nothing:
            return True
        phase = self.phase_map.get(action.__name__, '')
        host_view = self.observation.obs[target_host]
        # print("Host view:", host_view)
        if phase == "pingsweep":  # valid if host hasn't been sweeped
            return not host_view["sweeped"]
        elif phase == "portscan":  # valid if host hasn't been scanned and host has been sweeped
            return host_view["sweeped"] and not host_view["scanned"]
        elif phase == "discovery":  # valid if host has been scanned and sweeped but not discovered
            return host_view["sweeped"] and host_view["scanned"] and not host_view["discovered"]
        elif phase == "lateral-movement":  # valid if host has been scanned and sweeped and discovered but not on target
            return (
                host_view["sweeped"]
                and host_view["scanned"]
                and host_view["discovered"]
                and not host_view["on_host"]
            )
        elif phase == "privilege-escalation":  # valid if host has been scanned and sweeped and discovered and on target but not escalated
            return (
                host_view["sweeped"]
                and host_view["scanned"]
                and host_view["discovered"]
                and host_view["on_host"]
                and not host_view["escalated"]
            )
        elif phase == "impact":  # valid if host has been scanned and sweeped and discovered and on target and escalated
            return (
                host_view["sweeped"]
                and host_view["scanned"]
                and host_view["discovered"]
                and host_view["on_host"]
                and host_view["escalated"]
                and not host_view["impacted"]
            )
        else:
            return False


    def get_reward_map(self) -> RewardMap:
        return self.reward_map

    def get_action_space_shape(self) -> tuple[int, ...]:
        return self.action_space.get_shape()

    def get_observation_space(self):
        """
        Takes red agent view of network and transforms it into the obs vector.
        """
        return np.array(self.observation.obs_vec, dtype=np.int64)

    def reset(self, network: Network, service_mapping: dict) -> Iterable:
        super().reset(network, service_mapping)

        self.action_space.reset(self.current_host.name)
        self.current_step = 0
        return self.observation.reset(network, self.current_host.name)
