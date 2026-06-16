# EVRPTW route simulator, constructive/HC/ILS solvers, and parallel baseline job helper.

import csv
import hashlib
import json
import os
import time

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EV_SPECS_PATH = os.path.join(_SCRIPT_DIR, "..", "ev_specs.json")

# Iteration budgets: baseline_fast=True uses HC_BASELINE_ITERS for CSV sweeps.
_HC_BASELINE_ITERS = 36
_HC_FAST_ITERS = 80
_HC_QUALITY_ITERS = 500
_ILS_FAST_OUTER = 12
_ILS_FAST_KICK = 3
_ILS_FAST_INNER = 18
_ILS_QUALITY_OUTER = 35
_ILS_QUALITY_KICK = 5
_ILS_QUALITY_INNER = 40


# Load battery_capacity and energy_consumption_rate from ev_specs.json.
def load_ev_specs(path=None):
    path = path or EV_SPECS_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return float(data["battery_capacity"]), float(data["energy_consumption_rate"])


# Parse Homberger .txt: fleet, load cap, depot tuple, customer tuples, station tuples (id > 1000).
def parse_instance(filepath):
    lines = open(filepath, encoding="utf-8").readlines()
    vehicle_line = lines[4].split()
    fleet_size = int(vehicle_line[0])
    load_capacity = int(vehicle_line[1])

    depot = None
    customers = []
    stations = []

    for line in lines[9:]:
        parts = line.split()
        if len(parts) < 7:
            continue
        node_id = int(parts[0])
        x, y = int(parts[1]), int(parts[2])
        demand = int(parts[3])
        ready_time = int(parts[4])
        due_date = int(parts[5])
        service_time = int(parts[6])

        if node_id == 0:
            depot = (x, y, demand, ready_time, due_date, service_time)
        elif node_id <= 1000:
            customers.append((node_id, x, y, demand, ready_time, due_date, service_time))
        else:
            stations.append((node_id, x, y, demand, ready_time, due_date, service_time))

    return fleet_size, load_capacity, depot, customers, stations


# Map station_id -> charging_rate from *_stations.csv.
def load_station_rates(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {
            int(row["station_id"]): float(row["charging_rate"])
            for row in csv.DictReader(f)
        }


# NumPy distance matrix and arrays for customers/stations; station charging rates aligned to station order.
def prepare_instance(fleet_size, load_capacity, depot, customers, stations, station_rates):
    num_customers = len(customers)
    num_stations = len(stations)
    total_nodes = 1 + num_customers + num_stations

    coordinates = np.empty((total_nodes, 2), dtype=np.float64)
    coordinates[0] = [depot[0], depot[1]]

    customer_ready_time = np.empty(num_customers, dtype=np.float64)
    customer_due_date = np.empty(num_customers, dtype=np.float64)
    customer_demand = np.empty(num_customers, dtype=np.float64)
    customer_service_time = np.empty(num_customers, dtype=np.float64)

    for i, (_nid, x, y, dem, rdy, due, svc) in enumerate(customers):
        coordinates[1 + i] = [x, y]
        customer_ready_time[i] = rdy
        customer_due_date[i] = due
        customer_demand[i] = dem
        customer_service_time[i] = svc

    charging_rate = np.empty(num_stations, dtype=np.float64)
    for i, (nid, x, y, *_rest) in enumerate(stations):
        coordinates[1 + num_customers + i] = [x, y]
        charging_rate[i] = station_rates.get(nid, 1.0)

    diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    distance_matrix = np.sqrt(np.sum(diff ** 2, axis=2))

    station_indices = np.arange(1 + num_customers, total_nodes)
    station_to_depot_dist = distance_matrix[station_indices, 0]

    return {
        "fleet_size": fleet_size,
        "load_capacity": float(load_capacity),
        "num_customers": num_customers,
        "depot_closing_time": float(depot[4]),
        "distance_matrix": distance_matrix,
        "customer_indices": np.arange(1, 1 + num_customers),
        "station_indices": station_indices,
        "customer_ready_time": customer_ready_time,
        "customer_due_date": customer_due_date,
        "customer_demand": customer_demand,
        "customer_service_time": customer_service_time,
        "charging_rate": charging_rate,
        "station_to_depot_dist": station_to_depot_dist,
    }


# Simulate one open route (local customer indices): TW, load, battery with full recharges at nearest stations, return to depot.
def _simulate_route(
    route_local,
    instance,
    Q,
    h,
    max_recharge_legs=500,
):
    if not route_local:
        return {
            "feasible": True,
            "distance": 0.0,
            "charges": 0,
            "end_node": 0,
            "end_battery": float(Q),
            "end_clock": 0.0,
            "end_load": 0.0,
        }

    dist = instance["distance_matrix"]
    station_indices = instance["station_indices"]
    charging_rate = instance["charging_rate"]
    station_to_depot = instance["station_to_depot_dist"]
    ready = instance["customer_ready_time"]
    due = instance["customer_due_date"]
    dem = instance["customer_demand"]
    svc = instance["customer_service_time"]
    load_cap = instance["load_capacity"]
    t_close = instance["depot_closing_time"]

    current = 0
    battery = float(Q)
    clock_t = 0.0
    load = 0.0
    total_dist = 0.0
    charges = 0
    recharge_legs = 0

    def go_to_station_for_energy(target_customer_node):
        nonlocal current, battery, clock_t, total_dist, charges, recharge_legs
        while h * dist[current, target_customer_node] > battery + 1e-9:
            d_to_s = dist[current, station_indices]
            ok = h * d_to_s <= battery
            if not np.any(ok):
                return False
            si = int(np.argmin(np.where(ok, d_to_s, np.inf)))
            leg = float(d_to_s[si])
            total_dist += leg
            battery -= h * leg
            need = Q - battery
            clock_t += leg
            if need > 0:
                clock_t += need / float(charging_rate[si])
            battery = Q
            current = int(station_indices[si])
            charges += 1
            recharge_legs += 1
            if recharge_legs > max_recharge_legs:
                return False
        return True

    for loc in route_local:
        target = int(1 + loc)
        if load + dem[loc] > load_cap + 1e-9:
            return {"feasible": False, "distance": 0.0, "charges": charges}

        if not go_to_station_for_energy(target):
            return {"feasible": False, "distance": 0.0, "charges": charges}

        leg = float(dist[current, target])
        total_dist += leg
        battery -= h * leg
        arrival = clock_t + leg
        start = max(arrival, ready[loc])
        if start > due[loc] + 1e-9:
            return {"feasible": False, "distance": 0.0, "charges": charges}
        if start + svc[loc] > t_close + 1e-9:
            return {"feasible": False, "distance": 0.0, "charges": charges}

        clock_t = start + svc[loc]
        load += dem[loc]
        current = target

    if not go_to_station_for_energy(0):
        return {"feasible": False, "distance": 0.0, "charges": charges}

    leg_back = float(dist[current, 0])
    total_dist += leg_back
    battery -= h * leg_back
    clock_t += leg_back
    if clock_t > t_close + 1e-9:
        return {"feasible": False, "distance": 0.0, "charges": charges}

    return {
        "feasible": True,
        "distance": total_dist,
        "charges": charges,
        "end_node": int(current),
        "end_battery": float(battery),
        "end_clock": float(clock_t),
        "end_load": float(load),
    }


# Check partition feasibility: one visit per customer, fleet limit, each route simulated by _simulate_route.
def evaluate_routes(routes, instance, Q, h):
    fleet = instance["fleet_size"]
    n = instance["num_customers"]
    routes = [list(r) for r in routes if r]
    if len(routes) > fleet:
        return {"feasible": False, "total_distance": float("inf"), "num_routes": 0, "num_charges": 0}

    seen = set()
    for r in routes:
        for c in r:
            if c in seen or c < 0 or c >= n:
                return {"feasible": False, "total_distance": float("inf"), "num_routes": 0, "num_charges": 0}
            seen.add(c)

    if len(seen) != n:
        return {"feasible": False, "total_distance": float("inf"), "num_routes": 0, "num_charges": 0}

    total_d = 0.0
    total_ch = 0
    for r in routes:
        sim = _simulate_route(r, instance, Q, h)
        if not sim["feasible"]:
            return {"feasible": False, "total_distance": float("inf"), "num_routes": 0, "num_charges": 0}
        total_d += sim["distance"]
        total_ch += sim["charges"]

    return {
        "feasible": True,
        "total_distance": round(total_d, 2),
        "num_routes": len(routes),
        "num_charges": total_ch,
    }


# Greedy multi-route construction with optional RCL; repair pass inserts unvisited customers where feasible.
def constructive_solve_with_routes(
    instance,
    battery_capacity,
    energy_rate,
    rng=None,
    rcl_size=None,
):
    fleet_size = instance["fleet_size"]
    num_customers = instance["num_customers"]
    depot_closing_time = instance["depot_closing_time"]

    Q = float(battery_capacity)
    h = float(energy_rate)

    unvisited = np.ones(num_customers, dtype=bool)
    num_unvisited = num_customers
    routes = []

    while num_unvisited > 0 and len(routes) < fleet_size:
        current_node = 0
        battery = Q
        current_time = 0.0
        current_load = 0.0
        current_route_local = []

        while current_time < depot_closing_time:
            unvisited_positions = np.where(unvisited)[0]
            if len(unvisited_positions) == 0:
                break

            found_customer = False
            sim_matched = []
            for ui in np.where(unvisited)[0]:
                cust_idx = int(ui)
                trial_route = current_route_local + [cust_idx]
                sr = _simulate_route(trial_route, instance, Q, h)
                if sr["feasible"]:
                    sim_matched.append((cust_idx, sr))

            if sim_matched:
                if rng is not None and rcl_size is not None:
                    sim_matched.sort(key=lambda t: float(t[1]["end_clock"]))
                    k = min(int(rcl_size), len(sim_matched))
                    cust_idx, sim_res = sim_matched[int(rng.integers(0, k))]
                else:
                    cust_idx, sim_res = min(
                        sim_matched, key=lambda t: float(t[1]["end_clock"])
                    )

                current_route_local.append(cust_idx)
                current_node = int(sim_res["end_node"])
                battery = float(sim_res["end_battery"])
                current_time = float(sim_res["end_clock"])
                current_load = float(sim_res["end_load"])
                unvisited[cust_idx] = False
                num_unvisited -= 1
                found_customer = True

            if not found_customer:
                if current_route_local:
                    break
                if num_unvisited > 1:
                    uu = list(np.where(unvisited)[0])
                    pair_ok = False
                    for ii in range(len(uu)):
                        if pair_ok:
                            break
                        for jj in range(ii + 1, len(uu)):
                            a, b = int(uu[ii]), int(uu[jj])
                            for pair in ((a, b), (b, a)):
                                sr2 = _simulate_route(list(pair), instance, Q, h)
                                if sr2["feasible"]:
                                    current_route_local = list(pair)
                                    current_node = int(sr2["end_node"])
                                    battery = float(sr2["end_battery"])
                                    current_time = float(sr2["end_clock"])
                                    current_load = float(sr2["end_load"])
                                    unvisited[pair[0]] = False
                                    unvisited[pair[1]] = False
                                    num_unvisited -= 2
                                    found_customer = True
                                    pair_ok = True
                                    break
                            if pair_ok:
                                break
                    if pair_ok:
                        continue
                break

        if current_route_local:
            fin = _simulate_route(current_route_local, instance, Q, h)
            if not fin["feasible"]:
                for c in current_route_local:
                    unvisited[c] = True
                    num_unvisited += 1
                break
            routes.append(list(current_route_local))
        else:
            break

    changed = True
    while changed and num_unvisited > 0:
        changed = False
        for ui in list(np.where(unvisited)[0]):
            u = int(ui)
            placed = False
            for ri, r in enumerate(routes):
                rlist = list(r)
                for pos in range(len(rlist) + 1):
                    trial = rlist[:pos] + [u] + rlist[pos:]
                    sr = _simulate_route(trial, instance, Q, h)
                    if sr["feasible"]:
                        routes[ri] = trial
                        unvisited[u] = False
                        num_unvisited -= 1
                        changed = True
                        placed = True
                        break
                if placed:
                    break

    total_distance = 0.0
    total_charges = 0
    for r in routes:
        fin = _simulate_route(r, instance, Q, h)
        if fin["feasible"]:
            total_distance += float(fin["distance"])
            total_charges += int(fin["charges"])

    feasible = num_unvisited == 0
    ev = {
        "feasible": feasible,
        "total_distance": round(total_distance, 2) if feasible else float("inf"),
        "num_routes": len(routes),
        "num_charges": total_charges,
        "served": num_customers - num_unvisited,
    }
    return routes, ev


# Run construction then evaluate_routes; return feasible flag and route/charge stats (partial if infeasible).
def constructive_feasibility_summary(instance, battery_capacity, energy_rate):
    Q, h = float(battery_capacity), float(energy_rate)
    routes, ev0 = constructive_solve_with_routes(instance, Q, h)
    ev = evaluate_routes(routes, instance, Q, h)
    if ev["feasible"]:
        nr = int(ev["num_routes"])
        nc = int(ev["num_charges"])
        avg_cv = (nc / nr) if nr > 0 else 0.0
        return {
            "feasible": True,
            "num_routes": nr,
            "num_charges": nc,
            "avg_charges_per_vehicle": avg_cv,
            "served": int(instance["num_customers"]),
            "total_distance": ev["total_distance"],
        }
    partial_charges = 0
    for r in routes:
        if not r:
            continue
        sim = _simulate_route(r, instance, Q, h)
        if sim["feasible"]:
            partial_charges += int(sim["charges"])
        else:
            partial_charges += int(sim.get("charges", 0))
    nr = len(routes)
    avg_cv = (partial_charges / nr) if nr > 0 else 0.0
    return {
        "feasible": False,
        "num_routes": nr,
        "num_charges": partial_charges,
        "avg_charges_per_vehicle": avg_cv,
        "served": int(ev0["served"]),
        "total_distance": float("inf"),
    }


# Deep copy of route lists (local customer indices per route).
def _deep_copy_routes(routes):
    return [list(r) for r in routes]


# Random intra-route swap or inter-route relocate (structure only; may be infeasible).
def _random_neighbor_move(routes, rng):
    n_r = len(routes)
    if n_r == 0:
        return None, None

    if rng.random() < 0.55:
        candidates = [i for i, r in enumerate(routes) if len(r) >= 2]
        if not candidates:
            return None, None
        ri = int(rng.choice(candidates))
        trial = _deep_copy_routes(routes)
        r = trial[ri]
        a, b = rng.choice(len(r), size=2, replace=False)
        i, j = int(min(a, b)), int(max(a, b))
        r[i], r[j] = r[j], r[i]
        return trial, ("swap", ri, i, j)

    idx_from = [i for i, r in enumerate(routes) if len(r) > 0]
    if not idx_from:
        return None, None
    r_from = int(rng.choice(idx_from))
    pos = int(rng.integers(0, len(routes[r_from])))
    cust = int(routes[r_from][pos])
    r_to = int(rng.integers(0, n_r))
    pos_to = int(rng.integers(0, len(routes[r_to]) + 1))

    trial = _deep_copy_routes(routes)
    trial[r_from].pop(pos)
    if r_to == r_from and pos_to > pos:
        pos_to -= 1
    trial[r_to].insert(pos_to, cust)
    return trial, ("rel", cust, r_from, r_to)


# Sample _random_neighbor_move until evaluate_routes is feasible or max_attempts exhausted.
def _random_feasible_neighbor(routes, instance, Q, h, rng, max_attempts=50):
    for _ in range(max_attempts):
        trial, _rec = _random_neighbor_move(routes, rng)
        if trial is None:
            continue
        ev2 = evaluate_routes(trial, instance, Q, h)
        if ev2["feasible"]:
            return trial, ev2
    return None, None


# CON baseline: constructive_solve_with_routes + evaluate_routes, timed.
def constructive_only_solve(instance, battery_capacity, energy_rate):
    Q, h = float(battery_capacity), float(energy_rate)
    t0 = time.perf_counter()
    routes, ev0 = constructive_solve_with_routes(instance, Q, h)
    ev = evaluate_routes(routes, instance, Q, h)
    if not ev["feasible"]:
        out = {
            "feasible": False,
            "served": ev0.get("served", 0),
            "total_distance": float("inf"),
            "num_routes": len(routes),
            "num_charges": ev0.get("num_charges", 0),
            "method": "CON",
        }
        out["seconds"] = round(time.perf_counter() - t0, 4)
        return out
    out = {
        "feasible": True,
        "served": int(instance["num_customers"]),
        "total_distance": ev["total_distance"],
        "num_routes": ev["num_routes"],
        "num_charges": ev["num_charges"],
        "method": "CON",
    }
    out["seconds"] = round(time.perf_counter() - t0, 4)
    return out


# HC: start from construction; random swap/relocate neighbors that improve total distance.
def hill_climb_solve(instance, battery_capacity, energy_rate, max_iterations, seed):
    rng = np.random.default_rng(seed)
    Q, h = float(battery_capacity), float(energy_rate)
    t0 = time.perf_counter()

    routes, ev0 = constructive_solve_with_routes(instance, Q, h)
    ev_cur = evaluate_routes(routes, instance, Q, h)
    if not ev_cur["feasible"]:
        out = {
            "feasible": False,
            "served": ev0.get("served", 0),
            "total_distance": float("inf"),
            "num_routes": len(routes),
            "num_charges": ev0.get("num_charges", 0),
            "method": "HC",
        }
        out["seconds"] = round(time.perf_counter() - t0, 4)
        return out

    current_routes = _deep_copy_routes(routes)
    current_cost = ev_cur["total_distance"]
    best_ev = dict(ev_cur)

    for _ in range(max_iterations):
        trial, _rec = _random_neighbor_move(current_routes, rng)
        if trial is None:
            continue
        ev2 = evaluate_routes(trial, instance, Q, h)
        if not ev2["feasible"]:
            continue
        if ev2["total_distance"] < current_cost:
            current_routes = trial
            current_cost = ev2["total_distance"]
            if current_cost < best_ev["total_distance"]:
                best_ev = dict(ev2)

    out = {
        "feasible": best_ev["feasible"],
        "served": instance["num_customers"] if best_ev["feasible"] else best_ev.get("served", 0),
        "total_distance": best_ev["total_distance"],
        "num_routes": best_ev["num_routes"],
        "num_charges": best_ev["num_charges"],
        "method": "HC",
    }
    out["seconds"] = round(time.perf_counter() - t0, 4)
    return out


# ILS: perturb best with feasible random moves, short inner HC, accept improving global best.
def ils_solve(
    instance,
    battery_capacity,
    energy_rate,
    n_outer,
    kick_moves,
    inner_hc_iters,
    seed,
):
    rng = np.random.default_rng(seed)
    Q, h = float(battery_capacity), float(energy_rate)
    t0 = time.perf_counter()

    routes, ev0 = constructive_solve_with_routes(instance, Q, h)
    ev_cur = evaluate_routes(routes, instance, Q, h)
    if not ev_cur["feasible"]:
        out = {
            "feasible": False,
            "served": ev0.get("served", 0),
            "total_distance": float("inf"),
            "num_routes": len(routes),
            "num_charges": ev0.get("num_charges", 0),
            "method": "ILS",
        }
        out["seconds"] = round(time.perf_counter() - t0, 4)
        return out

    best_routes = _deep_copy_routes(routes)
    best_ev = dict(ev_cur)

    for _ in range(n_outer):
        work = _deep_copy_routes(best_routes)
        for _ in range(kick_moves):
            trial, ev_t = _random_feasible_neighbor(work, instance, Q, h, rng)
            if trial is not None:
                work = trial

        ev_w = evaluate_routes(work, instance, Q, h)
        if not ev_w["feasible"]:
            continue
        current = work
        cost = ev_w["total_distance"]
        for _ in range(inner_hc_iters):
            trial, _rec = _random_neighbor_move(current, rng)
            if trial is None:
                continue
            ev2 = evaluate_routes(trial, instance, Q, h)
            if ev2["feasible"] and ev2["total_distance"] < cost:
                current = trial
                cost = ev2["total_distance"]

        ev_end = evaluate_routes(current, instance, Q, h)
        if ev_end["feasible"] and ev_end["total_distance"] < best_ev["total_distance"]:
            best_routes = _deep_copy_routes(current)
            best_ev = dict(ev_end)

    out = {
        "feasible": best_ev["feasible"],
        "served": instance["num_customers"] if best_ev["feasible"] else best_ev.get("served", 0),
        "total_distance": best_ev["total_distance"],
        "num_routes": best_ev["num_routes"],
        "num_charges": best_ev["num_charges"],
        "method": "ILS",
    }
    out["seconds"] = round(time.perf_counter() - t0, 4)
    return out


# Deterministic RNG seed from instance name, method, optional salt and Q.
def _stable_seed(instance_name, method_label, salt=0, battery_q=None):
    parts = [str(instance_name), str(method_label), str(salt)]
    if battery_q is not None:
        parts.append(str(float(battery_q)))
    s = "|".join(parts)
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16) % (2**31 - 1)


# Copy of instance with fleet_size capped; None leaves instance unchanged.
def instance_with_fleet_cap(instance, fleet_cap):
    if fleet_cap is None:
        return instance
    file_fleet = int(instance["fleet_size"])
    eff = min(file_fleet, int(fleet_cap))
    if eff == file_fleet:
        return instance
    out = dict(instance)
    out["fleet_size"] = eff
    return out


# Dispatch CON / HC / ILS with iteration budgets from fast / baseline_fast flags; optional Q,h from ev_specs.
def run_meta_solver(
    instance,
    method_label,
    instance_name="",
    battery_capacity=None,
    energy_rate=None,
    fleet_cap=None,
    fast=True,
    baseline_fast=False,
):
    inst = instance_with_fleet_cap(instance, fleet_cap)

    if battery_capacity is None or energy_rate is None:
        q, hh = load_ev_specs()
        if battery_capacity is None:
            battery_capacity = q
        if energy_rate is None:
            energy_rate = hh

    q_seed = float(battery_capacity)
    if baseline_fast:
        hc_n = _HC_BASELINE_ITERS
        ils_o, ils_k, ils_i = 4, 2, 10
    elif fast:
        hc_n = _HC_FAST_ITERS
        ils_o, ils_k, ils_i = _ILS_FAST_OUTER, _ILS_FAST_KICK, _ILS_FAST_INNER
    else:
        hc_n = _HC_QUALITY_ITERS
        ils_o, ils_k, ils_i = _ILS_QUALITY_OUTER, _ILS_QUALITY_KICK, _ILS_QUALITY_INNER

    t_wall = time.perf_counter()

    if method_label == "CON":
        out = constructive_only_solve(inst, battery_capacity, energy_rate)
    elif method_label == "HC":
        out = hill_climb_solve(
            inst,
            battery_capacity,
            energy_rate,
            max_iterations=hc_n,
            seed=_stable_seed(instance_name, method_label, 0, q_seed),
        )
    elif method_label == "ILS":
        out = ils_solve(
            inst,
            battery_capacity,
            energy_rate,
            n_outer=ils_o,
            kick_moves=ils_k,
            inner_hc_iters=ils_i,
            seed=_stable_seed(instance_name, method_label, 1, q_seed),
        )
    else:
        raise ValueError("Unknown method_label: " + str(method_label))

    out["seconds"] = round(time.perf_counter() - t_wall, 4)
    return out


# One baseline run for multiprocessing: parse instance, run_meta_solver(..., baseline_fast=True), return sort key + CSV row.
def run_baseline_job(task):
    variant, base, txt_path, csv_stations, Q, method_label, energy_rate, sort_idx = task
    fleet_size, load_capacity, depot, customers, stations = parse_instance(txt_path)
    instance = prepare_instance(
        fleet_size,
        load_capacity,
        depot,
        customers,
        stations,
        load_station_rates(csv_stations),
    )
    fleet_in_file = int(instance["fleet_size"])
    n_cust = int(instance["num_customers"])

    result = run_meta_solver(
        instance,
        method_label,
        instance_name=f"{variant}_{base}",
        battery_capacity=float(Q),
        energy_rate=energy_rate,
        fleet_cap=None,
        fast=True,
        baseline_fast=True,
    )
    feas = bool(result["feasible"])
    dist = result["total_distance"]
    nr = int(result["num_routes"])
    nchg = int(result["num_charges"])

    if feas and dist != float("inf") and nr > 0:
        dist_out = dist
        n_veh = nr
        pct = round(100.0 * nr / float(fleet_in_file), 2)
        avg_cv = round(nchg / nr, 4)
    else:
        dist_out = ""
        n_veh = ""
        pct = ""
        avg_cv = ""

    row = [
        variant,
        base,
        int(Q),
        energy_rate,
        method_label,
        fleet_in_file,
        1 if feas else 0,
        result["served"],
        n_cust,
        dist_out,
        nr,
        n_veh,
        pct,
        nchg,
        avg_cv,
        result["seconds"],
    ]
    sort_key = (variant, base, int(Q), int(sort_idx))
    return sort_key, row
