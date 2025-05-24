import pandas as pd
import networkx as nx
import folium
import webbrowser
import math
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from geopy.distance import geodesic
import heapq
from collections import defaultdict
from collections import deque
import numpy as np

# load SCATS coordinates and neighbours
df_map = pd.read_csv("data\\map.csv")
df_map['SCATS Number'] = df_map['SCATS Number'].astype(int)

# shift scats' coordinates slightly since the original lat and lon in the data set dont map google map correctly
scats_coords = {
    int(row['SCATS Number']): (row['Latitude'] + 0.0012, row['Longitude'] + 0.0012)
    for _, row in df_map.iterrows()
}

# Create display mapping for SCATS number and location
scats_display_map = {
    int(row['SCATS Number']): f"{int(row['SCATS Number'])} - {row['Site Description']}"
    for _, row in df_map.iterrows()
}
display_to_scats = {v: k for k, v in scats_display_map.items()}

# load traffic volume data
# (predictions made by BiLSTM model are used to estimate speed and travel time since BiLSTM performs the best out of 3 models)
df_data = pd.read_csv("complete_oct_nov_csv\\bilstm_model\\bilstm_model_complete_data.csv")
df_data['SCATS Number'] = df_data['SCATS Number'].astype(int)
df_data['Date'] = pd.to_datetime(df_data['Date'])
traffic_avg = df_data.groupby(['SCATS Number', 'Date', 'time_of_day'])['traffic_volume'].mean().reset_index()
traffic_dict = defaultdict(dict)
for _, row in traffic_avg.iterrows():
    date_str = row['Date'].strftime('%Y-%m-%d')  # date to string
    if date_str not in traffic_dict[row['SCATS Number']]:
        traffic_dict[row['SCATS Number']][date_str] = {}
    traffic_dict[row['SCATS Number']][date_str][row['time_of_day']] = row['traffic_volume']

# unique time and date options
time_options = sorted(df_data['time_of_day'].unique())
date_options = sorted(df_data['Date'].dt.strftime('%Y-%m-%d').unique())

def compute_distance_km(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

# graph structure
G = nx.DiGraph()
for _, row in df_map.iterrows():
    A = int(row['SCATS Number'])
    if pd.notna(row['Neighbours']):
        neighbors = [int(n.strip()) for n in str(row['Neighbours']).split(';')]
        for B in neighbors:
            if B in scats_coords:
                dist = compute_distance_km(scats_coords[A][0], scats_coords[A][1], 
                                            scats_coords[B][0], scats_coords[B][1])
                G.add_edge(A, B, distance=dist)

# travel time estimator functions
def estimate_speed_from_flow(flow):
    a = -1.4648375  # coefficient for traffic_volume
    b = 93.75    # coefficient for traffic_volume
    c = -flow
    d = b * b - (4 * a * c)
    speed = (-b - math.sqrt(d)) / (2 * a)
    speed = min(speed, 60)  # capped speed: 60 km/h
    speed = max(speed, 5)  # minimum speed: 5 km/h
    return speed

def calculate_travel_time(u, v, date, time):
    distance = G[u][v]['distance']
    flow = traffic_dict[u][date][time]
    speed = estimate_speed_from_flow(flow)
    # add 30 seconds for intersection delay, then convert to minute
    travel_time = (distance / speed) * 60 + 30 / 60
    return travel_time

# search algorithms from assignment 2A
def dfs_search(edges, origin, destinations):
    stack = [(origin, [origin])]
    visited = set()
    
    while stack:
        current, path = stack.pop()
        if current in destinations:
            return path, len(visited)
        if current not in visited:
            visited.add(current)
            for neighbor, _ in reversed(edges.get(current, [])):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))
    return [], len(visited)

def bfs_search(edges, origin, destinations):
    queue = deque([[origin]])
    visited = set()
    
    while queue:
        path = queue.popleft()
        current = path[-1]
        if current in destinations:
            return path, len(visited)
        if current not in visited:
            visited.add(current)
            for neighbor, _ in edges.get(current, []):
                if neighbor not in visited:
                    queue.append(path + [neighbor])
    return [], len(visited)

def depth_limited_search(edges, origin, destinations, limit):
    stack = [(origin, [origin], 0)]
    visited = set()

    while stack:
        current, path, depth = stack.pop()
        if current in destinations:
            return path, len(visited)
        if current not in visited:
            visited.add(current)
            if depth < limit:
                # push children in reverse order so that the first neighbor is expanded first
                for neighbor, _ in reversed(edges.get(current, [])):
                    if neighbor not in visited:
                        stack.append((neighbor, path + [neighbor], depth + 1))
    return [], len(visited)

def iterative_deepening_search(edges, origin, destinations, max_depth=10):
    last_count = 0
    for depth in range(max_depth):
        path, count = depth_limited_search(edges, origin, destinations, depth)
        last_count = count
        if path:
            return path, count
    return [], last_count

def gbfs(edges, origin, destinations, heuristic):
    def get_best_heuristic(node):
        return min(heuristic[node][dest] for dest in destinations)

    pq = [(get_best_heuristic(origin), origin)]
    expanded = []
    parent = {origin: None}

    while pq:
        _, current = heapq.heappop(pq)

        if current in destinations:
            expanded.append(current)
            return build_path(current, parent), len(expanded)

        if current not in expanded:
            expanded.append(current)
            for neighbor, _ in edges.get(current, []):
                if neighbor not in expanded and all(neighbor != n for _, n in pq):
                    heapq.heappush(pq, (get_best_heuristic(neighbor), neighbor))
                    parent[neighbor] = current

    return [], len(expanded)

def a_star(edges, origin, destinations, heuristic):
    def get_best_heuristic(node):
        return min(heuristic[node][dest] for dest in destinations)

    g_cost = {origin: 0}
    f_cost = {origin: get_best_heuristic(origin)}
    pq = [(f_cost[origin], origin)]
    expanded = []
    parent = {origin: None}

    while pq:
        _, current = heapq.heappop(pq)

        if current in destinations:
            expanded.append(current)
            return build_path(current, parent), len(expanded)

        if current not in expanded:
            expanded.append(current)
            for neighbor, cost in edges.get(current, []):
                new_g = g_cost[current] + cost
                if neighbor not in g_cost or new_g < g_cost[neighbor]:
                    g_cost[neighbor] = new_g
                    f_cost[neighbor] = new_g + get_best_heuristic(neighbor)
                    heapq.heappush(pq, (f_cost[neighbor], neighbor))
                    parent[neighbor] = current

    return [], len(expanded)

def weighted_a_star(edges, origin, destinations, heuristic, weight=1.5):
    def get_best_heuristic(node):
        return min(heuristic[node][dest] for dest in destinations)
    
    g_cost = {origin: 0}
    f_cost = {origin: get_best_heuristic(origin) * weight}
    pq = [(f_cost[origin], origin)]
    expanded = []
    parent = {origin: None}

    while pq:
        _, current = heapq.heappop(pq)

        if current in destinations:
            expanded.append(current)
            return build_path(current, parent), len(expanded)

        if current not in expanded:
            expanded.append(current)
            for neighbor, cost in edges.get(current, []):
                new_g = g_cost[current] + cost
                if neighbor not in g_cost or new_g < g_cost[neighbor]:
                    g_cost[neighbor] = new_g
                    f_cost[neighbor] = new_g + weight * get_best_heuristic(neighbor)
                    heapq.heappush(pq, (f_cost[neighbor], neighbor))
                    parent[neighbor] = current

    return [], len(expanded)

def build_path(node, parent):
    path = []
    while node is not None:
        path.append(node)
        node = parent.get(node)
    return path[::-1]

def calculate_route_details(path, edges, date, start_time):
    total_time = 0
    total_distance = 0
    steps = []

    # Convert start_time (e.g., "07:00") to minutes since midnight
    h, m = map(int, start_time.split(":"))
    current_minutes = h * 60 + m
    current_date = pd.to_datetime(date)

    for i in range(len(path)-1):
        u, v = path[i], path[i+1]

        # Round down to nearest 15-minute interval
        rounded_minutes = (int(current_minutes) // 15) * 15
        time_of_day = f"{rounded_minutes//60:02d}:{rounded_minutes%60:02d}"

        # Calculate travel time for this segment
        travel_time = calculate_travel_time(u, v, date, time_of_day)
        total_time += travel_time

        # Update current_minutes for next segment
        current_minutes += travel_time
        # If current_minutes >= 1440, increment date and wrap around
        if current_minutes >= 1440:
            current_minutes -= 1440
            current_date += pd.Timedelta(days=1)
            date = current_date.strftime('%Y-%m-%d')

        dist = compute_distance_km(*scats_coords[u], *scats_coords[v])
        total_distance += dist

        steps.append({
            'from': u,
            'to': v,
            'time': f"{travel_time:.2f} min",
            'distance': f"{dist:.2f} km"
        })

    return {
        'total_time': f"{total_time:.2f} min",
        'total_distance': f"{total_distance:.2f} km",
        'steps': steps
    }

def improved_heuristic(node, dest_coords, traffic_data):
    base_dist = compute_distance_km(*scats_coords[node], *dest_coords)
    
    # Adjust based on historical traffic
    avg_speed = 50  # km/h, default
    if node in traffic_data:
        avg_flow = np.mean(list(traffic_data[node].values()))
        avg_speed = max(50 - (avg_flow / 100), 20)
    
    return (base_dist / avg_speed) * 60

# map functions
def _create_base_map(df_map, zoom_start=13):
    # create a base folium map with the center calculated from all SCATS sites
    center = [
        df_map['Latitude'].mean(),
        df_map['Longitude'].mean()
    ]
    return folium.Map(location=center, zoom_start=zoom_start)

def _add_all_sites(m, scats_coords, color='blue', radius=10, opacity=0.7):
    # add all SCATS sites to the map with specified style
    for scats, coord in scats_coords.items():
        folium.CircleMarker(
            location=coord,
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=opacity,
            popup=f"SCATS {scats}"
        ).add_to(m)

    # connections between scats (blue lines)
    for u, v in G.edges:
        lat1, lon1 = scats_coords[u]
        lat2, lon2 = scats_coords[v]
        folium.PolyLine([(lat1, lon1), (lat2, lon2)], color="blue", weight=1).add_to(m)

    return m

# GUI
def show_routes():
    src_value = combo_src.get()
    dst_value = combo_dst.get()
    time = combo_time.get()
    date = combo_date.get()
    algorithm = combo_algorithm.get()
    if not src_value or not dst_value or not time or not date or not algorithm:
        messagebox.showerror(title="Missing input", message="Please fill in all fields")
        return
    src = display_to_scats[src_value]
    dst = display_to_scats[dst_value]

    # build dynamic edges with time-dependent costs
    edges = defaultdict(list)
    for u, v in G.edges():
        travel_time = calculate_travel_time(u, v, date, time)
        edges[u].append((v, travel_time))

    # calculate heuristic
    dest_coords = scats_coords[dst]
    heuristic = defaultdict(dict)
    for node in G.nodes():
        node_coords = scats_coords[node]
        dist = compute_distance_km(*node_coords, *dest_coords)
        heuristic[node][dst] = (dist / 64) * 60  # maximum speed heuristic

    # algorithm selection
    path = []
    nodes_expanded = 0
    
    if algorithm == "A* Search":
        path, nodes_expanded = a_star(edges, src, [dst], heuristic)
    elif algorithm == "Depth First Search":
        path, nodes_expanded = dfs_search(edges, src, [dst])
    elif algorithm == "Breadth First Search":
        path, nodes_expanded = bfs_search(edges, src, [dst])
    elif algorithm == "Greedy Best First Search":
        path, nodes_expanded = gbfs(edges, src, [dst], heuristic)
    elif algorithm == "Depth-Limited Search":
        path, nodes_expanded = iterative_deepening_search(edges, src, [dst])
    elif algorithm == "Weighted A* Search":
        path, nodes_expanded = weighted_a_star(edges, src, [dst], heuristic, 1.5)
    
    # update display
    route_display.delete("1.0", tk.END)
    if not path:
        route_display.insert(tk.END, "No path found")
    else:
        details = calculate_route_details(path, edges, date, time)
        route_display.insert(tk.END, f"Algorithm: {algorithm}\n")
        route_display.insert(tk.END, f"Nodes expanded: {nodes_expanded}\n")
        route_display.insert(tk.END, f"Total Time: {details['total_time']}\n")
        route_display.insert(tk.END, f"Total Distance: {details['total_distance']}\n\n")
        
        for step in details['steps']:
            from_display = scats_display_map.get(step['from'], str(step['from']))
            to_display = scats_display_map.get(step['to'], str(step['to']))
            route_display.insert(
                tk.END,
                f"From {from_display} to {to_display}\n"
                f"Time: {step['time']} | Distance: {step['distance']}\n\n"
    )
        
        # draw map
        map = _create_base_map(df_map, zoom_start=13)
        map = _add_all_sites(map, scats_coords, color='blue', radius=10, opacity=0.7)
        
        # highlight the route
        path_coords = [scats_coords[n] for n in path]
        folium.PolyLine(path_coords, color='red', weight=5).add_to(map)
        
        # save and display map
        map.save("map.html")
        webbrowser.open("map.html")

# GUI layout
root = tk.Tk()
root.title("SCATS Route Finder")
root.geometry("1100x700")

left_frame = ttk.Frame(root, padding=10)
left_frame.grid(row=0, column=0, sticky="nsew")
root.grid_columnconfigure(0, minsize=400)

# origin scats input
ttk.Label(left_frame, text="Origin SCATS:").grid(row=0, column=0, sticky="w")
combo_src = ttk.Combobox(left_frame, values=sorted(scats_display_map.values()), state="readonly", width=45)
combo_src.grid(row=0, column=1, padx=(20, 0))

# destination scats input
ttk.Label(left_frame, text="Destination SCATS:").grid(row=1, column=0, sticky="w")
combo_dst = ttk.Combobox(left_frame, values=sorted(scats_display_map.values()), state="readonly", width=45)
combo_dst.grid(row=1, column=1, padx=(20, 0))

# date input
ttk.Label(left_frame, text="Date:").grid(row=2, column=0, sticky="w")
combo_date = ttk.Combobox(left_frame, values=date_options, state="readonly", width=45)
combo_date.grid(row=2, column=1, padx=(20, 0))
combo_date.set("2006-11-01")

# start time
ttk.Label(left_frame, text="Start Time:").grid(row=3, column=0, sticky="w")
combo_time = ttk.Combobox(left_frame, values=time_options, state="readonly", width=45)
combo_time.grid(row=3, column=1, padx=(20, 0))

# add algorithm selection
ttk.Label(left_frame, text="Algorithm:").grid(row=4, column=0, sticky="w")
algorithms = ["A* Search", "Depth First Search", "Breadth First Search", "Greedy Best First Search", "Depth-Limited Search", "Weighted A* Search"]
combo_algorithm = ttk.Combobox(left_frame, values=algorithms, state="readonly", width=45)
combo_algorithm.grid(row=4, column=1, padx=(20, 0))
combo_algorithm.set("A* Search")

# find route
ttk.Button(left_frame, text="Find Route", command=show_routes).grid(row=5, columnspan=2, pady=10)
status = tk.StringVar()
ttk.Label(left_frame, textvariable=status).grid(row=6, columnspan=2)

right_frame = ttk.Frame(root, padding=10)
right_frame.grid(row=0, column=1, sticky="nsew")

route_display = tk.Text(right_frame, wrap=tk.WORD, width=79, height=42)
route_display.insert(tk.END, "Route details will be displayed here.\n")
route_display.pack(fill=tk.BOTH, expand=True)

root.mainloop()