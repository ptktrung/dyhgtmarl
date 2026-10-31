import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# ==========================================
# 1. CONFIGURATION
# ==========================================
NUM_TRANSACTIONS = 100000  # Increased to 100k for "Big Data" requirement
START_TIME = datetime(2025, 1, 1, 6, 0, 0)

# Define Graph Nodes (Vertices)
# Core: District 1 (High density, good connectivity)
# Periphery: District 12 (Lower density, spotty connectivity)
NODES = {
    'Core': [f'BusNode_{i:02d}' for i in range(1, 11)] + [f'MetroNode_{i:02d}' for i in range(1, 6)],
    'Periphery': [f'BusNode_{i:02d}' for i in range(11, 21)] + [f'FerryNode_{i:02d}' for i in range(1, 4)]
}

# Graph Edges (Simulated adjacency for cascading faults)
# Node i affects Node i+1
ADJACENCY_MAP = {}
all_nodes = NODES['Core'] + NODES['Periphery']
for i in range(len(all_nodes) - 1):
    ADJACENCY_MAP[all_nodes[i]] = all_nodes[i+1]

def generate_iot_sensing_data(num_rows):
    """
    Generates synthetic IoT sensor data with graph-theoretic fault dependencies.
    """
    data = []
    current_time = START_TIME
    
    # State tracking for cascading faults
    previous_node_fault = False
    
    print(f"Generating {num_rows} transactions...")
    
    for i in range(num_rows):
        # 1. Spatiotemporal Attributes
        time_increment = random.randint(1, 15) # Random seconds between pings
        current_time += timedelta(seconds=time_increment)
        
        # Select Zone and Node
        zone_type = np.random.choice(['Core', 'Periphery'], p=[0.6, 0.4])
        node_id = np.random.choice(NODES[zone_type])
        mode = 'Metro' if 'Metro' in node_id else ('Ferry' if 'Ferry' in node_id else 'Bus')
        
        # 2. IoT Telemetry Generation (Normal vs. Anomaly)
        # Core has better signal (-50 to -70 dBm), Periphery is worse (-80 to -100 dBm)
        base_signal = -60 if zone_type == 'Core' else -85
        signal_noise = np.random.normal(0, 5)
        signal_strength = base_signal + signal_noise
        
        # Latency: Normal is <50ms. 
        base_latency = 20 if mode == 'Metro' else 45
        latency = max(5, np.random.normal(base_latency, 10))
        
        # Drift: GPS drift or Clock sync error (0 is perfect)
        drift = 0.0
        
        # 3. Fault Injection & Graph Propagation
        is_error = False
        error_type = "None"
        
        # Probability of error increases if:
        # a) Signal is weak (< -90)
        # b) Previous node in graph had a fault (Cascading Effect)
        prob_failure = 0.02
        if signal_strength < -90: prob_failure += 0.15
        if previous_node_fault: prob_failure += 0.20 # Graph dependency!
        
        if random.random() < prob_failure:
            is_error = True
            previous_node_fault = True # Propagate to next transaction
            
            # Classify Fault
            rand_fault = random.random()
            if rand_fault < 0.33:
                error_type = "Signal_Loss"
                signal_strength = -120 # Dead zone
            elif rand_fault < 0.66:
                error_type = "High_Latency"
                latency += random.randint(200, 2000) # Spike
            else:
                error_type = "Sensor_Drift"
                drift = random.uniform(5.0, 50.0) # Meters or Seconds deviation
        else:
            previous_node_fault = False
            
        # 4. Transaction Data
        fare = 12000 if mode == 'Metro' else (15000 if mode == 'Ferry' else 7000)
        
        data.append([
            f"TRX-{100000+i}", 
            current_time, 
            node_id,
            zone_type, 
            mode, 
            round(signal_strength, 2), 
            round(latency, 2), 
            round(drift, 4),
            fare, 
            is_error, 
            error_type
        ])

    columns = [
        "TransactionID", "Timestamp", "NodeID", "Zone", "Mode", 
        "SignalStrength_dBm", "Latency_ms", "Drift_Magnitude", 
        "Fare", "Is_Anomaly", "Fault_Type"
    ]
    return pd.DataFrame(data, columns=columns)

# ==========================================
# 2. EXECUTION & SAVE
# ==========================================
df = generate_iot_sensing_data(NUM_TRANSACTIONS)

# Save to CSV for your Paper
filename = "HCMC_IoT_Graph_Sensor_Data.csv"
df.to_csv(filename, index=False)

print(f"\nDataset generated: {filename}")
print(df.head())
print("\nFault Distribution:")
print(df['Fault_Type'].value_counts())