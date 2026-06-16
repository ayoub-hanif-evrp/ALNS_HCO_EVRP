# Homberger-format I/O, paths, density, and log-normal helpers for dataset generation.

import math
import os
import json


INPUT_DIR = '../homberger_1000_customer_instances'
VARIANT_A_DIR = '../variant_A_original_TW'
VARIANT_B_DIR = '../variant_B_relaxed_TW'
PARAMS_FILE = '../params.json'


# Euclidean distance between two planar points.
def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


# Load params.json next to the repo (relative path PARAMS_FILE).
def load_params():
    with open(PARAMS_FILE, 'r') as f:
        return json.load(f)


# Parse a Homberger .txt: return instance name, vehicle line dict, depot node, and customer list (no stations).
def parse_instance(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    instance_name = lines[0].strip()
    
    # line 4: NUMBER CAPACITY
    vehicle_line = lines[4].split()
    vehicle_info = {
        'number': int(vehicle_line[0]),
        'capacity': int(vehicle_line[1])
    }
    
    # customers start at line 9 (after header)
    depot = None
    customers = []
    
    for line in lines[9:]:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) < 7:
            continue
        
        node = {
            'id': int(parts[0]),
            'x': int(parts[1]),
            'y': int(parts[2]),
            'demand': int(parts[3]),
            'ready_time': int(parts[4]),
            'due_date': int(parts[5]),
            'service_time': int(parts[6])
        }
        
        if node['id'] == 0:
            depot = node
        else:
            customers.append(node)
    
    return instance_name, vehicle_info, depot, customers


# Write Homberger-format instance with depot, customers, then appended station nodes.
def write_instance(filepath, instance_name, vehicle_info, depot, customers, stations):
    with open(filepath, 'w') as f:
        f.write(instance_name + '\n')
        f.write('\n')
        f.write('VEHICLE\n')
        f.write('NUMBER     CAPACITY\n')
        f.write(' {}          {}\n'.format(vehicle_info['number'], vehicle_info['capacity']))
        f.write('\n')
        f.write('CUSTOMER\n')
        f.write('CUST NO.  XCOORD.    YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE TIME\n')
        f.write(' \n')
        
        # write depot
        f.write(format_node(depot))
        
        # write customers
        for c in customers:
            f.write(format_node(c))
        
        # write stations
        for s in stations:
            f.write(format_node(s))


# One formatted CUSTOMER line for the Homberger text format.
def format_node(node):
    return "{:>5} {:>9} {:>10} {:>10} {:>10} {:>10} {:>10}\n".format(
        node['id'],
        node['x'],
        node['y'],
        node['demand'],
        node['ready_time'],
        node['due_date'],
        node['service_time']
    )


# Sorted list of .txt filenames in a folder.
def get_instance_files(folder_path):
    files = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith('.txt'):
            files.append(fname)
    return sorted(files)


# For each customer id, count other customers within Euclidean distance R.
def compute_customer_density(customers, R):
    densities = {}
    for i, ci in enumerate(customers):
        count = 0
        for j, cj in enumerate(customers):
            if i != j:
                dist = euclidean_distance(ci['x'], ci['y'], cj['x'], cj['y'])
                if dist <= R:
                    count += 1
        densities[ci['id']] = count
    return densities


# Map target mean and std of a log-normal to (mu, sigma) parameters.
def compute_lognormal_params(desired_mean, desired_std):
    mean_sq = desired_mean ** 2
    var = desired_std ** 2
    
    sigma_sq = math.log(1 + var / mean_sq)
    sigma = math.sqrt(sigma_sq)
    mu = math.log(desired_mean) - sigma_sq / 2
    
    return mu, sigma