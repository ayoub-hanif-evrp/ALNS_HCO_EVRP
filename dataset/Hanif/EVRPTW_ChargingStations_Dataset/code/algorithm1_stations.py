# Generate charging stations and station CSVs from Homberger instances into variant_A_original_TW (params.json).

import random
import os
import csv
from utils import (
    euclidean_distance, parse_instance, write_instance, load_params, 
    get_instance_files, compute_customer_density, compute_lognormal_params,
    INPUT_DIR, VARIANT_A_DIR
)


# Sample station (x,y) in the customer bbox with minimum pairwise separation d_sep (fallback if max attempts).
def generate_station_locations(x_min, x_max, y_min, y_max, num_stations, d_sep, max_attempts=1000):
    stations_xy = []
    
    for _ in range(num_stations):
        placed = False
        for _ in range(max_attempts):
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            
            too_close = False
            for sx, sy in stations_xy:
                if euclidean_distance(x, y, sx, sy) < d_sep:
                    too_close = True
                    break
            
            if not too_close:
                stations_xy.append((x, y))
                placed = True
                break
        
        if not placed:
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            stations_xy.append((x, y))
    
    return stations_xy


# For each station, average local customer density over K nearest customers, then normalize to [0,1] (or 0.5 if flat).
def compute_station_density_scores(stations_xy, customers, customer_densities, K):
    scores = []
    
    for sx, sy in stations_xy:
        dists = []
        for c in customers:
            d = euclidean_distance(sx, sy, c['x'], c['y'])
            dists.append((d, c['id']))
        
        dists.sort(key=lambda x: x[0])
        nearest_ids = [cid for _, cid in dists[:K]]
        
        avg_dens = sum(customer_densities[cid] for cid in nearest_ids) / len(nearest_ids)
        scores.append(avg_dens)
    
    d_low = min(scores)
    d_high = max(scores)
    
    if d_high == d_low:
        return [0.5] * len(scores)
    
    return [(s - d_low) / (d_high - d_low) for s in scores]


# Build station nodes and attribute rows (rates, log-normal price/wait params, density_score) from params.
def generate_stations(depot, customers, params):
    random.seed(params['seed'])
    
    num_stations = params['num_stations']
    d_sep = params['d_sep']
    r_min, r_max = params['r_min'], params['r_max']
    p_min, p_max = params['p_min'], params['p_max']
    w_min, w_max = params['w_min'], params['w_max']
    K = params['K']
    R_factor = params['R_factor']
    cv_price = params['cv_price']
    cv_wait = params['cv_wait']
    
    xs = [c['x'] for c in customers]
    ys = [c['y'] for c in customers]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    R = R_factor * max(x_max - x_min, y_max - y_min)
    
    customer_densities = compute_customer_density(customers, R)
    stations_xy = generate_station_locations(x_min, x_max, y_min, y_max, num_stations, d_sep)
    density_scores = compute_station_density_scores(stations_xy, customers, customer_densities, K)
    
    stations = []
    attributes = []
    
    # station IDs start after last customer
    start_id = max(c['id'] for c in customers) + 1
    
    for idx, ((sx, sy), d_s) in enumerate(zip(stations_xy, density_scores)):
        station_id = start_id + idx
        
        charging_rate = r_min + (r_max - r_min) * d_s
        
        price_mean = p_min + (p_max - p_min) * d_s
        price_mean = max(price_mean, 0.01)
        price_std = cv_price * price_mean
        price_mu, price_sigma = compute_lognormal_params(price_mean, price_std)
        
        wait_mean = w_min + (w_max - w_min) * d_s
        wait_mean = max(wait_mean, 0.01)
        wait_std = cv_wait * wait_mean
        wait_mu, wait_sigma = compute_lognormal_params(wait_mean, wait_std)
        
        station = {
            'id': station_id,
            'x': int(round(sx)),
            'y': int(round(sy)),
            'demand': 0,
            'ready_time': 0,
            'due_date': depot['due_date'],
            'service_time': 0
        }
        stations.append(station)
        
        attr = {
            'station_id': station_id,
            'charging_rate': round(charging_rate, 4),
            'price_mu': round(price_mu, 4),
            'price_sigma': round(price_sigma, 4),
            'wait_mu': round(wait_mu, 4),
            'wait_sigma': round(wait_sigma, 4),
            'density_score': round(d_s, 4)
        }
        attributes.append(attr)
    
    return stations, attributes


# Write *_stations.csv with the standard attribute columns.
def write_stations_csv(filepath, attributes):
    fieldnames = ['station_id', 'charging_rate', 'price_mu', 'price_sigma', 'wait_mu', 'wait_sigma', 'density_score']
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for attr in attributes:
            writer.writerow(attr)


# Process all Homberger .txt files under INPUT_DIR and write augmented instances + CSVs to VARIANT_A_DIR.
def main():
    params = load_params()
    os.makedirs(VARIANT_A_DIR, exist_ok=True)
    
    files = get_instance_files(INPUT_DIR)
    print("Processing {} instances from {}".format(len(files), INPUT_DIR))
    
    for fname in files:
        name = os.path.splitext(fname)[0]
        fpath = os.path.join(INPUT_DIR, fname)
        
        instance_name, vehicle_info, depot, customers = parse_instance(fpath)
        stations, attributes = generate_stations(depot, customers, params)
        
        out_txt = os.path.join(VARIANT_A_DIR, name + '.txt')
        out_csv = os.path.join(VARIANT_A_DIR, name + '_stations.csv')
        
        write_instance(out_txt, instance_name, vehicle_info, depot, customers, stations)
        write_stations_csv(out_csv, attributes)
        
        print("  {} -> {} customers, {} stations".format(fname, len(customers), len(stations)))
    
    print("Done. Output in {}".format(VARIANT_A_DIR))


if __name__ == '__main__':
    main()