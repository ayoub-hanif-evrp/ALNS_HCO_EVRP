# Copy variant A to variant B with distance-based customer due-date relaxation (params.json).

import os
import math
import shutil
from utils import load_params, euclidean_distance, VARIANT_A_DIR, VARIANT_B_DIR


# For each .txt in input_folder, cap-extend customer due dates by near/medium/far bands; copy other files as-is.
def relax_time_windows(input_folder, output_folder, R_near, R_medium, R_far):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        # skip hidden folders and checkpoints
        if filename.startswith('.') or 'checkpoint' in filename:
            continue
        
        src = os.path.join(input_folder, filename)
        
        # skip directories
        if os.path.isdir(src):
            continue
        
        if not filename.lower().endswith('.txt'):
            # copy non-txt files (csv)
            dst = os.path.join(output_folder, filename)
            shutil.copy(src, dst)
            continue
        
        filepath = os.path.join(input_folder, filename)
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # find depot (id = 0) coordinates and closing time
        depot_x, depot_y, depot_close = None, None, None
        max_dist = 0
        
        for line in lines[9:]:
            parts = line.split()
            if len(parts) >= 7 and parts[0] == '0':
                depot_x = int(parts[1])
                depot_y = int(parts[2])
                depot_close = int(parts[5])
                break
        
        if depot_x is None:
            continue
        
        # find max distance from customers (not stations)
        for line in lines[9:]:
            parts = line.split()
            if len(parts) >= 7:
                node_id = int(parts[0])
                if node_id > 0 and node_id <= 1000:  # customers only
                    x, y = int(parts[1]), int(parts[2])
                    dist = euclidean_distance(x, y, depot_x, depot_y)
                    max_dist = max(max_dist, dist)
        
        # process lines
        modified_lines = lines[:9]  # keep header
        
        for line in lines[9:]:
            parts = line.split()
            if len(parts) < 7:
                modified_lines.append(line)
                continue
            
            node_id = int(parts[0])
            
            # only relax customers (id 1-1000), not depot or stations
            if 1 <= node_id <= 1000:
                x, y = int(parts[1]), int(parts[2])
                due_date = int(parts[5])
                
                dist = euclidean_distance(x, y, depot_x, depot_y)
                
                if dist < max_dist / 3:
                    relaxation = R_near
                elif dist < 2 * max_dist / 3:
                    relaxation = R_medium
                else:
                    relaxation = R_far
                
                new_due_date = min(due_date + relaxation, depot_close)
                parts[5] = str(new_due_date)
            
            modified_line = "{:>5} {:>9} {:>10} {:>10} {:>10} {:>10} {:>10}\n".format(
                parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6]
            )
            modified_lines.append(modified_line)
        
        # write to output
        out_path = os.path.join(output_folder, filename)
        with open(out_path, 'w') as f:
            f.writelines(modified_lines)
        
        print("  {} -> relaxed".format(filename))


# Load R_near/R_medium/R_far from params and run relax_time_windows from VARIANT_A_DIR to VARIANT_B_DIR.
def main():
    params = load_params()
    
    R_near = params['R_near']
    R_medium = params['R_medium']
    R_far = params['R_far']
    
    print("Processing instances from {}".format(VARIANT_A_DIR))
    relax_time_windows(VARIANT_A_DIR, VARIANT_B_DIR, R_near, R_medium, R_far)
    print("Done. Output in {}".format(VARIANT_B_DIR))


if __name__ == '__main__':
    main()