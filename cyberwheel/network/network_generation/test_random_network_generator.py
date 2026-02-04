import random
from cyberwheel.network.network_generation.network_generator import NetworkYAMLGenerator

class RandomNetworkGenerator:
    """
    Generates random network topologies for Cyberwheel environment.
    Useful for RL training to avoid overfitting to specific network configurations.
    """
    
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
    
    def generate_random_network(
        self, 
        num_subnets=(2, 5),
        hosts_per_subnet=(3, 8),
        host_types=None,
        network_name=None
    ):
        """
        Generate a random network configuration.
        
        Args:
            num_subnets: tuple (min, max) for number of subnets
            hosts_per_subnet: tuple (min, max) for hosts per subnet
            host_types: list of host types to randomly assign. If None, uses defaults
            network_name: custom network name, if None generates random
        
        Returns:
            NetworkYAMLGenerator instance with random topology
        """
        
        if host_types is None:
            host_types = ["workstation", "web_server"] # remove database_server since it is not defined in host definitions?
        
        # Generate network name if not provided
        if network_name is None:
            network_name = f"random-network-{self.rng.randint(1000, 9999)}"
        
        network = NetworkYAMLGenerator(network_name=network_name, desc="Randomly generated network")
        
        # Create router
        router_name = "core_router"
        network.router(router_name)
        # add_route_to_router?
        
        # Random number of subnets
        n_subnets = self.rng.randint(*num_subnets)
        
        # Create subnets
        subnet_names = []
        for i in range(n_subnets): # unique identifier
            subnet_name = f"subnet_{i}"
            subnet_names.append(subnet_name)
            
            # Generate IP range (using 192.168.x.0/24) 
            ip_range = f"192.168.{i}.0/24"
            # Private IP address ranges are reserved blocks (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
            # 10.0.0.0–10.255.255.255 (large networks), 
            # 172.16.0.0–172.31.255.255 (medium networks), and 
            # 192.168.0.0–192.168.255.255 (small networks)
            
            # Create subnet
            network.subnet(
                subnet_name,
                router_name=router_name,
                ip_range=ip_range,
                dns_server=f"192.168.{i}.1"
            )
            
            # Add firewall rules (allow inter-subnet traffic with some probability)
            if self.rng.random() > 0.3:  # 70% chance of open firewall
                # exclude last element in list to avoid self-rule
                for other_subnet in subnet_names[:-1]:
                    network.add_firewall_to_subnet(
                        subnet_name,
                        name=f"allow from {other_subnet}",
                        src=other_subnet,
                        dest="",
                        port=0,
                        protocol=""
                    )
        
        # Add hosts to each subnet
        host_id = 0
        for subnet_name in subnet_names:
            n_hosts = self.rng.randint(*hosts_per_subnet)
            
            for j in range(n_hosts):
                host_name = f"host_{host_id:03d}" # format: pad with leading zeros to ensure 3 digits
                host_type = self.rng.choice(host_types) # add probabilities?
                
                network.host(
                    host_name,
                    subnet=subnet_name,
                    type_=host_type
                )
                
                # Randomly add SSH firewall rule (50% chance)
                if self.rng.random() > 0.5:
                    network.add_firewall_to_host(
                        host_name,
                        name="allow SSH",
                        src = "",
                        dest = "",
                        port=22,
                        protocol = ""
                    )
                
                host_id += 1
        
        # Optionally add some random interfaces between hosts (lateral movement paths)
        all_hosts = [f"host_{i:03d}" for i in range(host_id)]
        n_connections = self.rng.randint(1, min(5, len(all_hosts) // 2))
        
        for _ in range(n_connections):
            if len(all_hosts) >= 2:
                host1, host2 = self.rng.sample(all_hosts, 2)
                network.interface(host1, host2)
        
        print(f"Generated random network '{network_name}' with {n_subnets} subnets and {host_id} hosts.")
        return network
    
    def generate_and_save(self, output_path="cyberwheel/data/configs/network", **kwargs):
        """
        Generate random network and save to YAML file.
        
        Args:
            output_path: directory where YAML will be saved
            **kwargs: passed to generate_random_network()
        
        Returns:
            Path to generated YAML file
        """
        network = self.generate_random_network(**kwargs)
        network.output_yaml(path=output_path)
        return f"{output_path}/{network.file_name}.yaml"


def generate_random_networks(n_networks=10, output_path="cyberwheel/data/configs/network", seed=None):
    """
    Generate multiple random network configurations.
    
    Args:
        n_networks: number of networks to generate
        output_path: directory to save YAMLs
        seed: random seed for reproducibility
    
    Returns:
        list of generated YAML file paths
    """
    generator = RandomNetworkGenerator(seed=seed)
    files = []
    
    for i in range(n_networks):
        # Vary the parameters for diversity
        num_subnets = (2, random.randint(3, 6))
        hosts_per_subnet = (3, random.randint(5, 12))
        
        file_path = generator.generate_and_save(
            output_path=output_path,
            num_subnets=num_subnets,
            hosts_per_subnet=hosts_per_subnet,
            network_name=f"random-network-{i:03d}"
        )
        files.append(file_path)
    
    return files


if __name__ == "__main__":
    # Example usage
    generator = RandomNetworkGenerator(seed=42)
    
    # Generate single network
    network = generator.generate_random_network(
        num_subnets=(3, 5),
        hosts_per_subnet=(4, 8)
    )
    
    # Save to file
    network.output_yaml(path="./")
    
    print(f"Generated network: {network.file_name}.yaml")
    
    # Or generate multiple networks at once
    # files = generate_random_networks(n_networks=5, output_path="cyberwheel/data/configs/network", seed=123)
    # print(f"Generated {len(files)} networks")
