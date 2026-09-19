import heapq
from xml.parsers.expat import errors
import numpy as np
import igl
import polyscope as ps
from scipy.optimize import brentq
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from pathlib import Path
import json

# debugger functions ------------------------------------------------------------------------------------------------------------------

def debug_render_region_pinches(rid, F, V, FR, RP, RP_next_maps, failing_edge = None):

    ps.remove_all_structures()

    # 1. Render the region
    region_faces = np.where(FR == rid)[0]

    if len(region_faces) > 0:

        F_region = F[region_faces]

        region_vids = np.unique(F_region)

        vid_to_local = {
            int(v): i for i, v in enumerate(region_vids)
        }

        F_local = np.array([
            [vid_to_local[int(v)] for v in face]
            for face in F_region
        ], dtype=int)

        V_region = V[region_vids]

        ps.register_surface_mesh(
            f"REGION_{rid}",
            V_region,
            F_local,
            smooth_shade=False
        )

    # 2. Render all pinched vertices
    pinched_vertices = RP[rid]

    if len(pinched_vertices) > 0:

        pinched_vertices = np.asarray(
            pinched_vertices,
            dtype=int
        )

        P = V[pinched_vertices]

        ps.register_point_cloud(
            f"PINCHED_{rid}",
            P
        )

    # 3. Render all incoming pinched edges
    incoming_points = []
    incoming_edges = []

    for edge in RP_next_maps[rid]:

        vid_in, vid_pinched = edge

        p0 = V[int(vid_in)]
        p1 = V[int(vid_pinched)]

        base = len(incoming_points)

        incoming_points.append(p0)
        incoming_points.append(p1)

        incoming_edges.append([base, base + 1])

    if incoming_points:

        incoming_points = np.asarray(incoming_points)
        incoming_edges = np.asarray(incoming_edges)

        ps.register_curve_network(
            f"PINCHED_INCOMING_{rid}",
            incoming_points,
            incoming_edges,
            radius=0.001
        )

    # 4. Render the EXACT failing edge
    if failing_edge is not None:

        vid_this, vid_next = failing_edge

        print("FAILING EDGE")
        print("region :", rid)
        print("edge   :", (vid_this, vid_next))
        print("p0     :", V[int(vid_this)])
        print("p1     :", V[int(vid_next)])

        points = np.asarray([
            V[int(vid_this)],
            V[int(vid_next)]
        ])

        edges = np.array([
            [0, 1]
        ], dtype=int)

        ps.register_curve_network(
            f"FAILING_EDGE_{vid_this}_{vid_next}",
            points,
            edges,
            radius=0.001
        )

        # Also show the two endpoint vertices
        ps.register_point_cloud(
            f"FAILING_ENDPOINTS",
            points,
            radius=0.002
        )

    # 5. Print RP map
    print(f"REGION {rid} PINCHED MAP")

    for edge, value in RP_next_maps[rid].items():

        vid_in, vid_pinched = edge
        vid_out, rid_in, rid_out = value

        print(
            f"{vid_in} -> {vid_pinched} "
            f"-> {vid_out} "
            f"| {rid_in} -> {rid_out}"
        )

    ps.show()
def show_naked_vertex(V, VF, VV, VE, index):
    ps.remove_all_structures()

    naked_vertices = [
        vid for vid in range(len(VF))
        if VF[vid] and VF[vid][-1] == -1
    ]

    print("Number of naked vertices:", len(naked_vertices))

    if index < 0 or index >= len(naked_vertices):
        raise IndexError(
            f"index {index} out of range; "
            f"there are {len(naked_vertices)} naked vertices"
        )

    vid = naked_vertices[index]

    print("\nSELECTED NAKED VERTEX")
    print("naked-list index:", index)
    print("vertex id:", vid)
    print("VF:", VF[vid])
    print("VV:", VV[vid])
    print("VE:", VE[vid])
    print("position:", V[vid])

    V_np = np.asarray(V)

    # Remove duplicate neighbors if any, while preserving order
    neighbors = list(dict.fromkeys(int(v) for v in VV[vid]))

    # Points:
    #   local point 0 = selected vertex
    #   local point 1... = ring neighbors
    points = np.vstack([
        V_np[vid],
        V_np[neighbors]
    ])

    # Edges
    edges = []

    n = len(neighbors)

    # Center -> every neighbor
    for i in range(n):
        edges.append([0, i + 1])

    # Neighbor -> next neighbor
    for i in range(n - 1):
        j = i + 1
        edges.append([i + 1, j + 1])

    edges = np.asarray(edges, dtype=int)

    # Register 1-ring
    ps.register_curve_network(
        f"naked_vertex_{index}_1ring",
        points,
        edges,
        radius=0.001
    )

    # Register center
    ps.register_point_cloud(
        f"naked_vertex_{index}_CENTER",
        V_np[[vid]],
        radius=0.002
    )

    # Register neighbors
    ps.register_point_cloud(
        f"naked_vertex_{index}_NEIGHBORS",
        V_np[neighbors],
        radius=0.002
    )
    ps.show()

    return vid
def show_region_loop_debug(V, F, FR, VF, RV, RA, rid, loop_id):

    ps.remove_all_structures()

    V = np.asarray(V)
    F = np.asarray(F)
    FR = np.asarray(FR)

    # Check region / loop
    if rid < 0 or rid >= len(RV):
        raise IndexError(f"Invalid region id: {rid}")

    if loop_id < 0 or loop_id >= len(RV[rid]):
        raise IndexError(
            f"Invalid loop id {loop_id} for region {rid}. "
            f"Number of loops = {len(RV[rid])}"
        )

    loop = np.asarray(RV[rid][loop_id], dtype=int)

    print("Region:", rid)
    print("Loop:", loop_id)
    print("Boundary:", loop)

    # 1. Selected region
    region_faces = np.where(FR == rid)[0]

    region_mesh = ps.register_surface_mesh(
        f"region_{rid}",
        V,
        F[region_faces]
    )

    region_mesh.set_edge_width(0.0)

    # 2. Find neighboring regions
    #    ONLY around this boundary loop
    neighboring_rids = set()

    for vid in loop:

        for fid in VF[int(vid)]:

            if fid == -1:
                continue

            nrid = int(FR[fid])

            if nrid != rid:
                neighboring_rids.add(nrid)

    print("Neighboring regions:", sorted(neighboring_rids))

    # 3. Show neighboring regions
    for nrid in neighboring_rids:

        neighbor_faces = np.where(FR == nrid)[0]

        neighbor_mesh = ps.register_surface_mesh(
            f"neighbor_region_{nrid}",
            V,
            F[neighbor_faces]
        )

        neighbor_mesh.set_edge_width(0.0)

    # 4. Show ONLY this boundary loop
    n = len(loop)

    if n >= 2:

        edges = np.column_stack([
            np.arange(n),
            np.roll(np.arange(n), -1)
        ])

        ps.register_curve_network(
            f"region_{rid}_loop_{loop_id}",
            V[loop],
            edges,
            radius=0.001
        )

    # 5. Get anchors of THIS loop
    anchor_indices = RA[rid][loop_id]

    anchor_vids = [
        int(RV[rid][loop_id][anchor_index])
        for anchor_index in anchor_indices
    ]

    print("Region:", rid)
    print("Loop:", loop_id)
    print("Anchor indices:", anchor_indices)
    print("Anchor vertex IDs:", anchor_vids)

    # 6. Render anchors ONLY
    if anchor_vids:
        n = len(anchor_vids)
        ps.register_point_cloud(
            f"region_{rid}_loop_{loop_id}_anchors",
            V[anchor_vids],
            radius=0.002
        )
        edges = np.column_stack([
                    np.arange(n),
                    np.roll(np.arange(n), -1)
                ])
        ps.register_curve_network(
            f"region_{rid}_loop_{loop_id}_anchors_edges",
            V[anchor_vids],
            edges,
            radius=0.001
        )

    ps.show()
def random_colors(k, FR):
    np.random.seed(72)
    colors = np.random.rand(k, 3)
    face_colors = np.zeros((len(FR), 3))
    for fid in range(len(FR)):
        rid = FR[fid]
        face_colors[fid] = colors[rid]

    return face_colors
def display_labeled_points(V, Number, name="seed_Points", seed=42):

    import numpy as np
    import polyscope as ps

    V = np.asarray(V, dtype=float)
    Number = np.asarray(Number)

    if len(V) != len(Number):
        raise ValueError(
            "V and Number must have the same length."
        )

    # Unique labels
    labels = np.unique(Number)

    # One deterministic random color per label
    rng = np.random.default_rng(seed)
    label_colors = rng.random((len(labels), 3))

    # Map labels -> colors
    label_to_color = {
        label: label_colors[i]
        for i, label in enumerate(labels)
    }

    colors = np.array([
        label_to_color[n]
        for n in Number
    ])

    # Register ONE point cloud
    cloud = ps.register_point_cloud(
        name,
        V,
        radius=0.005
    )

    # Assign per-point colors
    cloud.add_color_quantity(
        "Number",
        colors
    )

    return cloud
def region_planar_deviation_colors(V, F, FR, name="Region Deviation"):
    import numpy as np
    import polyscope as ps

    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=int)
    FR = np.asarray(FR)

    if len(F) != len(FR):
        raise ValueError("F and FR must have the same length.")

    # One deviation value for each face
    face_deviation = np.zeros(len(F), dtype=float)

    # Region IDs
    regions = np.unique(FR)

    for region in regions:

        # Faces in this region
        face_ids = np.where(FR == region)[0]

        # All unique vertices belonging to those faces
        vertex_ids = np.unique(F[face_ids])
        points = V[vertex_ids]

        if len(points) < 3:
            deviation = 0.0

        else:
            # 1. PCA plane fitting

            centroid = np.mean(points, axis=0)

            X = points - centroid

            # PCA using SVD
            _, _, vh = np.linalg.svd(
                X,
                full_matrices=False
            )

            # Smallest principal direction = plane normal
            normal = vh[-1]

            normal /= np.linalg.norm(normal)

            # 2. Maximum distance to fitted plane

            distances = np.abs(X @ normal)

            deviation = np.max(distances)

        # Same deviation for every face in this region
        face_deviation[face_ids] = deviation

    # 3. Map deviation to 0~255
    d_min = np.min(face_deviation)
    d_max = np.max(face_deviation)

    if d_max > d_min:

        value = (
            (face_deviation - d_min)
            / (d_max - d_min)
        )

    else:
        value = np.zeros_like(face_deviation)

    # 4. Green -> Red
    colors = np.zeros((len(F), 3), dtype=float)

    colors[:, 0] = value          # Red
    colors[:, 1] = 1.0 - value    # Green
    colors[:, 2] = 0.0             # Blue

    # 5. Polyscope
    mesh = ps.register_surface_mesh(
        name,
        V,
        F
    )

    mesh.add_scalar_quantity(
        "Planar Deviation bar",
        face_deviation,
        defined_on="faces",
        cmap="turbo",
        enabled=True,
        vminmax=(0.0, 2.0)
    )

    mesh.add_color_quantity(
        "Planar Deviation",
        colors,
        defined_on="faces"
    )

    return face_deviation

#--------------------------------------------------------------------------------------------------------------------------------------






# Preprocessing -----------------------------------------------------------------------------------------------------------------------

def load_mesh(path):
    V, F = igl.read_triangle_mesh(path)
    V, F, _, _ = igl.remove_unreferenced(V, F)
    if F.shape[0] == 0:
        raise ValueError(f"No faces found when reading '{path}'")
    F_oriented, _ = igl.bfs_orient(F)
    return V, F_oriented
# Compute per face g, area, normal, and face covariance matrix
def compute_face_properties(V, F):
    g = np.mean(V[F], axis=1)
    A = igl.doublearea(V, F) / 2.0
    FN = igl.per_face_normals(V, F) # Can be smoothed if needed

    # Need to be double checked
    Q_paper = np.array([[10,7,0],
                        [7,10,0],
                        [0,0,0]])
    
    Q_exact = np.array([[2,-1,0],
                        [-1,2,0],
                        [0,0,0]])

    
    # Compute covariance matrix
    M_cov = np.zeros((3,3))

    V1 = V[F[:, 0]]
    V2 = V[F[:, 1]] 
    V3 = V[F[:, 2]]
    
    e1 = V2 - V1
    e2 = V3 - V1

    M = np.zeros((len(e1), 3, 3))
    M[:, 0, :] = e1
    M[:, 1, :] = e2

    MTQM = M.transpose(0,2,1) @ Q_exact @ M
    ggT = g[:, :, None] @ g[:, None, :]
    FM = ((1 / 36) * MTQM + ggT)

    return g, A, FN, FM
# Vertice-face adjacency in counter-clockwise order (ended with naked edge -1) 
# This function assumes that there are no non-manifold edges, but allows non-manifold vertices at naked edges. 
def vertex_face_adjacency(V, F, FF):
    # Select starting neighbor face for each vertex
    V_f_start = [-2] * len(V)
    naked_vertices = {}
    for fid in range(len(F)):
        for eid_local, fid_neighbor in enumerate(FF[fid]):
            # edge starting vertex
            vid_local = eid_local
            vid = F[fid][vid_local]

            # Naked edge
            if fid_neighbor == -1: 
                if vid not in naked_vertices:
                    V_f_start[vid] = fid # High priority: modifying in place
                    naked_vertices[vid] = []
                else:
                    # This vertex has already been assigned, so this is a vertex where 2 naked loops touch. This is a non-manifold vertex.
                    print(f"Vertex {vid} is a non-manifold vertex where 2 naked loops touch.")
                    naked_vertices[vid].append(fid)
                    
            # Inner edge
            elif V_f_start[vid] == -2: # haven't been assigned yet
                V_f_start[vid] = fid

    # Adjacency lists
    # Vertex --- face 
    VF = [[] for _ in range(len(V))]

    # Vertex --- vertex (1-ring neighborhood)
    VV = [[] for _ in range(len(V))]

    # Vertex --- edgeID (edge ID at face)
    VE = [[] for _ in range(len(V))]

    for vid in range(len(V)):
        # isolated vertex
        if V_f_start[vid] == -2:
            VF[vid] = []
            VV[vid] = []
            VE[vid] = []
            continue

        fid_list = []
        vid_list = []
        eid_list = []

        fid_next = V_f_start[vid]
        counter = 0
        while True:
            fid_this = fid_next
            fid_list.append(fid_this)
            
            # In igl convention, eid = its starting vertex's local id at this face, so we can use the starting vertex to find the next vertex and previous edge.
            
            # Vertex local id at this face
            vid_local = np.where(F[fid_this] == vid)[0][0] # np.where() returns tuple

            # Next vertex at this face
            vid_local_next = (vid_local + 1) % 3
            vid_neighbor = F[fid_this][vid_local_next]
            vid_list.append(vid_neighbor)

            # Previous/this edge at this face
            eid_local = vid_local
            eid_list.append(eid_local)
            eid_local_prev = (eid_local + 2) % 3

            # neighboring face across previous edge
            fid_next = FF[fid_this][eid_local_prev]

            # loop end at starting face
            if fid_next == V_f_start[vid]:
                VF[vid] = fid_list
                VV[vid] = vid_list
                VE[vid] = eid_list
                break

            # loop end at naked edge
            if fid_next == -1:
                # Previous vertex at this face
                vid_local_prev = (vid_local + 2) % 3
                vid_neighbor = F[fid_this][vid_local_prev]
                vid_list.append(vid_neighbor)

                fid_list.append(-1)
                eid_list.append(-1)

                VF[vid] = fid_list
                VV[vid] = vid_list
                VE[vid] = eid_list
                break

            counter += 1
            if counter > 999:
                raise ValueError(vid)

    for vid, naked_fids in naked_vertices.items():
        for fid in naked_fids:
            fid_list = []
            vid_list = []
            eid_list = []

            fid_next = fid # start from the first naked face
            counter = 0
            while True:
                fid_this = fid_next
                fid_list.append(fid_this)
                
                # In igl convention, eid = its starting vertex's local id at this face, so we can use the starting vertex to find the next vertex and previous edge.
                
                # Vertex local id at this face
                vid_local = np.where(F[fid_this] == vid)[0][0] # np.where() returns tuple

                # Next vertex at this face
                vid_local_next = (vid_local + 1) % 3
                vid_neighbor = F[fid_this][vid_local_next]
                vid_list.append(vid_neighbor)

                # Previous/this edge at this face
                eid_local = vid_local
                eid_list.append(eid_local)
                eid_local_prev = (eid_local + 2) % 3

                # neighboring face across previous edge
                fid_next = FF[fid_this][eid_local_prev]

                # loop end at naked edge
                if fid_next == -1:
                    # Previous vertex at this face
                    vid_local_prev = (vid_local + 2) % 3
                    vid_neighbor = F[fid_this][vid_local_prev]
                    vid_list.append(vid_neighbor)

                    fid_list.append(-1)
                    eid_list.append(-1)

                    VF[vid].extend(fid_list)
                    VV[vid].extend(vid_list)
                    VE[vid].extend(eid_list)
                    break

                counter += 1
                if counter > 999:
                    raise ValueError(vid)
        
    return VF, VV, VE # These 3 outputs should be in the same structure
def export_region_mesh(V, F, FR, output_filepath):

    data = {
        "vertices": V.tolist(),
        "faces": F.tolist(),
        "face_regions": FR.tolist(),
    }

    filepath = Path(output_filepath)

    with filepath.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

#--------------------------------------------------------------------------------------------------------------------------------------






# Post processing ---------------------------------------------------------------------------------------------------------------------

# flood fill to rebuild the regions and remove disconnectivity.
def rebuild_region(FR, FF, FE = None):

    new_FR = np.full(len(FR), -1, dtype=np.int32)
    new_D = []
    R_map = [] # new region id to old region id

    # Select any remaining face to start flood fill
    new_k = 0
    while np.any(new_FR == -1):
        fid_start = np.where(new_FR == -1)[0][0]
        old_rid = FR[fid_start]
        new_rid = new_k
        new_D.append(0.0)
        R_map.append(old_rid)

        # Flood fill
        q = [fid_start]
        while q:
            fid = q.pop()
            new_FR[fid] = new_rid
            new_D[new_rid] += FE[fid] if FE is not None else 0.0

            for fid_neighbor in FF[fid]:
                if fid_neighbor == -1:
                    continue
                if FR[fid_neighbor] == old_rid and new_FR[fid_neighbor] == -1:
                    q.append(fid_neighbor)

        new_k += 1
        if new_k > 9999:
            raise ValueError(f"Flood fill fails due to infinite loop")

    return new_k, new_FR, new_D, R_map
# Compute the topology for regions (the hardest part)
def compute_region_topology(F, k, FR, FF, VF, VV):
# 1. Compute boundary vertices next-map

    # Next-map for region boundary vertices
    # Stored in dictionary[vid_start:(vid_end, rid_opp)] for loop tracing
    RV_next_maps = [{} for _ in range(k)] 

    for fid in range(F.shape[0]):
        rid = FR[fid]
        for eid, fid_neighbor in enumerate(FF[fid]):

            # Get region ID at this neighbor face
            if fid_neighbor == -1:
                # Naked edge
                rid_neighbor = -1
            else:
                # Inner edge
                rid_neighbor = FR[fid_neighbor]

            if rid_neighbor != rid:
                vid_start = F[fid, eid]
                vid_end   = F[fid, (eid + 1) % 3]

                RV_next_maps[rid][vid_start] = (vid_end, rid_neighbor)

    # PROBLEM: at pinched vertex, it may have multiple next-vertex
    # e.g., the same vertex may be added to dictionary many times while erasing the previous next-vertex

# 2. Compute pinched vertices next-map
   
    # Pinched vertices in each region
    RP = [[] for _ in range(k)]

    for vid in range(len(VF)):
        if VF[vid] == []: # skip isolated vertex
            continue
        
        # Vertex neighboring regions in order: AAA-BBB-CCCC-DD. 
        rid_list = []
        for fid_neighbor in VF[vid]:

            # Get region ID at this neighbor face
            rid_neighbor = FR[fid_neighbor] if fid_neighbor != -1 else -1

            if len(rid_list) == 0: 
                rid_list.append(rid_neighbor)

            elif rid_list[-1] != rid_neighbor:
                rid_list.append(rid_neighbor)

        # It may look like: A(AA)-B(BB)-C(CCC)-D(D)-A(A) (head to tail loop). 
        # The last region A doesn't appear twice, so we remove it.
        if len(rid_list) > 1 and rid_list[-1] == rid_list[0]:
            rid_list.pop()

        # Check region repetition: A(AA)-B(BB)-'C(C)'-D(DD)-'C(CC)'
        # In this case, region C is repeated, so region C is pinched at this vertex
        rid_set = set()
        for rid_neighbor in rid_list:
            if rid_neighbor == -1: # naked edge
                continue
            if rid_neighbor not in rid_set:
                rid_set.add(rid_neighbor)
            else:
                RP[rid_neighbor].append(vid) # Add pinched vertex to this region

    # Pinched vertex CCW-neighborhood next-map
    # Stored in dictionary [(vid_entry, vid_pinched): (vid_exit, rid_opp_entry, rid_opp_exit)]
    RP_next_maps = [{} for _ in range(k)]

    for rid in range(k):
        for vid_pinched in RP[rid]:
            p_fid_list = VF[vid_pinched]

            # For a pinched vertex V at region R:
            # If we traverse its neighbors in CCW order,
            # It enters R from right side, and leaves R from left side.
            
            neighbor_list = [] # neighbors in CCW order, each neighbor is a tuple (vid_neighbor, rid_neighbor)
            first_left_side_boundary_id = 0 # for pairing left-side and right-side boundary neighbors

            for i in range(len(p_fid_list)):
                fid_this = p_fid_list[i]
                fid_prev = p_fid_list[i - 1]
                rid_this = FR[fid_this] if fid_this != -1 else -1
                rid_prev = FR[fid_prev] if fid_prev != -1 else -1
                vid_p_neighbor = VV[vid_pinched][i]

                # right-side boundary neighbor:
                if rid_prev != rid and rid_this == rid:
                    # This is the first right-side boundary neighbor, so we need to find the first left-side boundary neighbor
                    if not neighbor_list: 
                        first_left_side_boundary_id = -1
                        
                    neighbor_list.append((vid_p_neighbor, rid_prev))

                # left-side boundary neighbor:
                elif rid_prev == rid and rid_this != rid:
                    neighbor_list.append((vid_p_neighbor, rid_this))

            # The boundary vertices tracing is reversed.
            # It enters a pinched vertex from a left-side boundary.
            # It leaves a pinched vertex from the next right-side boundary.
            
            # Pair the left-side and right-side boundary neighbors in order, and update the next-map for the pinched vertex
            for i in range(len(neighbor_list) // 2):

                ls_id = first_left_side_boundary_id + 2 * i
                rs_id = ls_id + 1

                vid_ls, rid_ls = neighbor_list[ls_id]
                vid_rs, rid_rs = neighbor_list[rs_id]
                
                RP_next_maps[rid][(vid_ls, vid_pinched)] = (vid_rs, rid_ls, rid_rs) # Update the next-map for the pinched vertex

            del RV_next_maps[rid][vid_pinched] # delete pinched vertex in boundary vertices next-map.

# 3. construct loop

    # Boundaries in counter-clockwise order. If its inner loop then the order is clockwise
    # Loops are stored in flat lists and ended (seperated) by -2.
    
    # Region boundary Vertices
    RV = [[] for _ in range(k)]
    # Region boundary Anchors: ID of anchor vertices in the boundary vertices list
    RA = [[] for _ in range(k)]
    # Region boundary Regions: matching the anchor vertices (anchor vertex = where the region starts)
    RR = [[] for _ in range(k)]
    
    # topology error counter
    topology_errors = 0
    
    for rid in range(k):

        # boundary vertices/regions
        vid_next_map = RV_next_maps[rid]
        pinched_vid_next_map = RP_next_maps[rid]

        # loop until clearance of all remaining boundary vertices 
        v_remaining = set(vid_next_map)
        
        # If all boundary vertices are cleared but pinched vertices are not, loop until clearance of all remaining pinched vertices
        p_remaining = set(pinched_vid_next_map) # pinched vertices represented by the entry edge

        counter = 0
        while v_remaining:

            # Get any vertex in the remaining vertices
            for vid_first in v_remaining:
                break

            vid_list = [] # list of boundary vertices in this loop
            rid_list = [] # list of neighboring regions in this loop
            pid_list = [] # list of pinched vertices in this loop

            # debugger
            visited = set()

            # Pointer tracing
            vid_next = vid_first
            while True:
                vid_this = vid_next

                vid_list.append(vid_this)
                
                vid_next, rid_opp_this = vid_next_map[vid_this]
                rid_list.append(rid_opp_this)

                # If next vertex is a pinched vertex: 
                # use pinched next-map to look up the secondary next vertex,
                # so it jumps over the pinched vertex
                while vid_next not in vid_next_map: # vid_next is a pinched vertex
                    
                    edge = (vid_this, vid_next)
                    vid_next_next, _, rid_opp_next = pinched_vid_next_map[edge]
                    print(
                        "PINCHED LOOKUP", rid,
                        "incoming vertex =", vid_this,
                        "vid_pinched =", vid_next,
                        "outgoing vertex =", vid_next_next
                    )

                    vid_list.append(vid_next)
                    rid_list.append(rid_opp_next)
                    pid_list.append(edge)
                    
                    vid_this = vid_next
                    vid_next = vid_next_next
                    
                    counter += 1
                    if counter > 99999:
                        raise ValueError(f"Loop tracing fails at '{rid}' due to infinite loop of pinched vertices")

                if vid_next == vid_first:
                    break

                counter += 1
                if counter > 99999:
                    raise ValueError(f"Loop tracing fails at '{rid}'")

            ra_list = [] # list of anchor vertices in this loop
            rr_list = [] # list of neighboring regions in this loop
            for i in range(len(rid_list)):
                if rid_list[i] != rid_list[i - 1]:
                    ra_list.append(i) # Add anchor vertex local id 
                    rr_list.append(rid_list[i]) # Add neighboring region 
                    
            if not rr_list:
                rr_list.append(rid_list[0]) # If all neighboring regions are the same, add the first one

            if len(rr_list) < 3:
                topology_errors += 1

            RV[rid].append(vid_list)
            RA[rid].append(ra_list)
            RR[rid].append(rr_list)

            v_remaining -= set(vid_list) 
            p_remaining -= set(pid_list) 

            counter += 1
            if counter > 99999:
                raise ValueError(f"Loop tracing fails at '{rid}'")

        # If all boundary vertices are cleared but pinched vertices are not, it means there are pinched vertices loops
        # It is unlikely to happen for a good mesh, but we still need to handle it for robustness.
        while p_remaining: 
            
            # Get any edge in the remaining pinched vertices
            for edge_first in p_remaining:
                break

            vid_list = [] # list of boundary vertices in this loop
            rid_list = [] # list of neighboring regions in this loop
            pid_list = [] # list of pinched vertices in this loop

            # Pointer tracing
            edge_next = edge_first
            while True:
                edge_this = edge_next
                pid_list.append(edge_this)

                vid_this = edge_this[0]
                vid_next = edge_this[1]
                vid_list.append(vid_this)

                vid_next_next, rid_opp_this, _ = pinched_vid_next_map[edge_this]
                rid_list.append(rid_opp_this)

                edge_next = (vid_next, vid_next_next)

                if edge_next == edge_first:
                    break

                counter += 1
                if counter > 99999:
                    raise ValueError(f"Loop tracing fails at '{rid}' due to infinite loop of pinched vertices")

            ra_list = [] # list of anchor vertices in this loop
            rr_list = [] # list of neighboring regions in this loop
            for i in range(len(rid_list)):
                if rid_list[i] != rid_list[i - 1]:
                    ra_list.append(i) # Add anchor vertex local id 
                    rr_list.append(rid_list[i]) # Add neighboring region 

            if not rr_list:
                rr_list.append(rid_list[0]) # If all neighboring regions are the same, add the first one

            if len(rr_list) < 3:
                topology_errors += 1

            RV[rid].append(vid_list)
            RA[rid].append(ra_list)
            RR[rid].append(rr_list)

            p_remaining -= set(pid_list) 

            counter += 1
            if counter > 99999:
                raise ValueError(f"Loop tracing fails at '{rid}'")

    return RV, RA, RR, topology_errors
# delete the region lying between less than 3 other regions and merge with the best neighboring region.
def delete_invalid_regions(FR, RR, A):
    # Calculate the area of each region
    areas = [float(0) for _ in range(len(RR))] 
    for rid in range(len(RR)):
        mask = FR == rid
        area = np.sum(A[mask])
        areas[rid] = area

    # merging buffer
    chunks = [] # record the region id chunks that need to be merged
    
    for rid in range(len(RR)):
        for loop_id in range(len(RR[rid])):

            # merge the loops that have <3 neighbor
            if len(RR[rid][loop_id]) < 3:

                # Find the neighboring region with the maximum area to merge with
                max_area = -1.0
                rid_best_neighbor = -1 # selected neighbor

                for rid_neighbor in RR[rid][loop_id]:
                    if rid_neighbor == -1:
                        continue

                    area = areas[rid_neighbor]
                    if area > max_area:
                        max_area = area
                        rid_best_neighbor = rid_neighbor

                if rid_best_neighbor == -1:
                    continue

                # Check if this region or the neighboring region has already been merged with other regions
                cid_neighbor = -1
                cid_this = -1
                for cid, chunk in enumerate(chunks):
                    if rid_best_neighbor in chunk:
                        cid_neighbor = cid
                    if rid in chunk:
                        cid_this = cid

                # If no region has been merged with other regions, create a new chunk
                if cid_neighbor == cid_this == -1:
                    chunks.append(set([rid, rid_best_neighbor]))

                # If only one region has been merged with other regions, add the other region to the same chunk
                elif cid_neighbor == -1:
                    chunks[cid_this].add(rid_best_neighbor)
                elif cid_this == -1:
                    chunks[cid_neighbor].add(rid)

                # If both regions have been merged with other 2 different region chunks, merge the two chunks
                elif cid_neighbor != cid_this:
                    chunks[cid_neighbor] = chunks[cid_neighbor].union(chunks[cid_this])
                    chunks[cid_this].clear()

    # Select the head of each chunk (the region with the maximum area in this chunk)
    chunk_heads = []
    for chunk in chunks:
        if not chunk:
            chunk_heads.append(-1)
            continue

        # Head is the region with the maximum area in this chunk
        max_area = -1.0
        head = -1
        for rid in chunk:
            if areas[rid] > max_area:
                max_area = areas[rid]
                head = rid
        chunk_heads.append(head)

    # Update the region IDs in FR to the head of the chunk
    new_FR = np.copy(FR)
    for chunk, head in zip(chunks, chunk_heads):
        if head == -1:
            continue
        for rid in chunk:
            if rid != head:
                new_FR[FR == rid] = head

    return new_FR

#--------------------------------------------------------------------------------------------------------------------------------------
    
    




# VPR (plane proxy) ===================================================================================================================

"""
VSA: Variational Shape Approximation
1. Flood: S, P ----> R, E
2. Proxy Fitting + Best Triangle Selection: R, E ----> S, P
3. Proxy insertion + deletion: S, P ----> S, P
repeat 1-3 until convergence
"""
def VPR(mode, path, init_seeds):

    # Load mesh
    V, F = load_mesh(path)
    dir = Path(path).parent
    filename = Path(path).stem
    output_filepath1 = dir / f"{filename}_Partitions_{mode}.json"
    output_filepath2 = dir / f"{filename}_Simplified_{mode}.json"
    output_filepath3 = dir / f"{filename}_History_{mode}.npy"
    output_filepath4 = dir / f"{filename}_Seeds_{mode}.npy"
    output_filepath5 = dir / f"{filename}_Order_{mode}.npy"
    output_filepath6 = dir / f"{filename}_PX_{mode}.npy"
    output_filepath7 = dir / f"{filename}_PN_{mode}.npy"

    # Adjacency Lists
    # Face-face adjacency
    FF, _ = igl.triangle_triangle_adjacency(F)
    # Vertice-face adjacency
    VF, VV, _ = vertex_face_adjacency(V, F, FF)

    ps.init()
    ps.set_ground_plane_mode("none")

    # Face Properties
    g, A, FN, FM = compute_face_properties(V, F)
    FN = face_normal_smooth(FN, FF, A, 15, 0.9)

    # list of face artificial weights
    _, _, PV1, PV2, _ = igl.principal_curvature(V,F,radius=5,useKring=False)
    W  = np.mean(K_remap[F], axis=1)


    # Proxies
    S = list(init_seeds) # list of seed fids for each proxy
    k = len(init_seeds) # number of proxies
    PN = list(FN[S]) # list of proxy normals
    PX = list(g[S]) # list of proxy centroids
    alpha = [0.0] * k 


    # Loop
    iterations = 250
    history = np.zeros((250, F.shape[0]), dtype=np.uint8)

    convergence_history_max = []
    convergence_history_mean = []

    topology_errors_counter = 0
    last_topology_errors = 0


    for iteration in range(iterations):
        # Compute regions
        k, D, FR, FE = flood(mode, V, F, FF, A, W, S, FN, PN, PX, alpha)
        k, FR, D, _ = rebuild_region(FR, FF, FE)
        history[iteration] = FR

        RV, RA, RR, topology_errors = compute_region_topology(F, k, FR, FF, VF, VV)
        
        # Update Proxies
        for rid in range(k):
            mask = FR == rid
            if mode == "L2":
                pn, px = vector_compute_proxy_L2(g, A, W, FM, mask)
            elif mode == "L21":
                pn, px = vector_compute_proxy_L21(g, A, W, FN, mask)
            elif mode == "H1":
                pn, px, alpha[rid] = vector_compute_proxy_H1(g, A, W, FN, FM, mask)

            PN[rid] = pn
            PX[rid] = px

        # merge and split regions
        if iteration != iterations-1 and iteration != 0:
            # Check for topology traps: 
            # if the number of topology errors does not decrease for a certain number of iterations, we need to handle it.
            if topology_errors >= last_topology_errors and topology_errors > 0:
                topology_errors_counter += 1
                print(f"Topology errors not decreasing for {topology_errors_counter} iterations")
            else:
                topology_errors_counter = 0

            # Teleportation:
            if iteration % 10 == 0:  
                ratio = 2.0 * (1.0 - (np.float64(iteration) / np.float64(iterations)))
                ratio = max(ratio, 0.5)  # Ensure ratio does not go below 0.5
                k, FR, PX, PN, alpha, executed = teleportation(mode, ratio, V, F, k, FR, FE, D, RR, PX, PN, alpha, g, A, W, FN, FM)
                if not executed and (iterations-iteration) > iterations / 10:
                    k, FR, PX, PN, alpha = greedy_split(mode, k, FR, FE, PX, PN, alpha, D, g, A, W, FN, FM, 2.0)

            # Topology Handling:
            elif topology_errors_counter >= iterations / 10:
                # This method may try to split a region only has one face, which may result in a proxy with no faces assigned to it.
                # So we need to clean up the lists later.
                k, FR, PX, PN, alpha = topology_handling(mode, VF, k, RV, RA, RR, FR, FM, FE, PN, PX, alpha, g, A, W, FN)
                print(f"Topology errors not decreasing for maximal iterations, applying topology handling")
                topology_errors_counter = 0

            last_topology_errors = topology_errors

        # Clean up: remove proxies with no faces assigned to them
        new_S = []
        new_PN = []
        new_PX = []
        new_alpha = []
        for rid in range(k):
            if np.any(FR == rid):

                if mode == "L2":
                    new_S.append(compute_best_triangle_L2(V, F, A, W, FR, PN, PX, rid))
                elif mode == "L21":
                    new_S.append(compute_best_triangle_L21(A, W, FN, FR, PN, rid))
                elif mode == "H1":
                    new_S.append(compute_best_triangle_H1(V, F, A, W, FN, FR, PN, PX, alpha, rid))

                new_PN.append(PN[rid])
                new_PX.append(PX[rid])
                new_alpha.append(alpha[rid] if mode == "H1" else 0.0)

        S = new_S
        PN = new_PN
        PX = new_PX
        alpha = new_alpha

        convergence_history_max.append(max(D))
        convergence_history_mean.append(sum(D)/k if k > 0 else 0)
        print(f"Iteration {iteration}: {k} proxies, max error: {max(D)}, mean error: {sum(D)/k if k > 0 else 0}, topology errors: {topology_errors}")

    
    n_V, n_F, n_FR, seeds_all, seeds_boundary, order_map = simplify_mesh(V, F, FR, FF, VV, VF, PX, PN, np.float64(0.00)) #0.25
    n_V, n_F, _, _ = igl.remove_unreferenced(n_V, n_F)

    n_FF, _ = igl.triangle_triangle_adjacency(n_F)

    # Every rebuild need a map to the original region id.
    # Its important because the indexing to plane proxy will change.
    n_k, n_FR, _, R_map = rebuild_region(n_FR, n_FF)

    try:
        n_VF, n_VV, _ = vertex_face_adjacency(V, n_F, n_FF)
        n_RV, n_RA, n_RR, n_topology_errors = compute_region_topology(n_F, n_k, n_FR, n_FF, n_VF, n_VV)
        print(f"After simplification: {n_k} proxies, topology errors: {n_topology_errors}")

        times = 0
        while n_topology_errors > 0:
            times += 1
            if times > 10:  # Prevent infinite loop
                print("Maximum repair iterations reached.")
                break
            _, n_A, _, _ = compute_face_properties(V, n_F)
            n_FR = delete_invalid_regions(n_FR, n_RR, n_A)
            n_RV, n_RA, n_RR, n_topology_errors = compute_region_topology(n_F, n_k, n_FR, n_FF, n_VF, n_VV)
            print(f"After {times} times repair: topology errors: {n_topology_errors}")
            
        
        # Polygons
        if n_topology_errors == 0:
            n_PX = [PX[rid] for rid in R_map]
            n_PN = [PN[rid] for rid in R_map]

            np.save(output_filepath6, n_PX)
            np.save(output_filepath7, n_PN)
            
            polygons = compute_polygons(n_PX, n_PN, n_RR, n_V, n_RV, n_RA)

            # render polygons
            vertices, faces = triangulate_polygons_cdt(polygons)
            ps.register_surface_mesh("triangulated_polygons", vertices, faces)

            # Show polygons debugger
            all_points = []
            all_edges = []
            offset = 0
            for rid, polygon in enumerate(polygons):
                for loop_id, polyline in enumerate(polygon):
                    if len(polyline) < 2:
                        continue
                    points = np.asarray(polyline, dtype=np.float64)
                    n = len(points)
                    all_points.append(points)
                    # Closed loop
                    edges = np.column_stack([np.arange(n) + offset, np.arange(n) + 1 + offset])
                    # Last -> first
                    edges[-1, 1] = offset
                    all_edges.append(edges)
                    offset += n
            if all_points:
                all_points = np.vstack(all_points)
                all_edges = np.vstack(all_edges).astype(np.int32)
                ps.register_curve_network("all_polygons", all_points, all_edges, radius=0.002, color=(0.0, 0.0, 0.0))
        

    except ValueError as e:
        bad_vid = e.args[0]
        print(f"Non-manifold edge generated at vertex {bad_vid} after simplification. Please check the mesh and try again.")
        ps.register_point_cloud(
            "problematic_vertex",
            V[[bad_vid]],
            radius=0.01,
            color=[1.0, 0.0, 0.0]
        )
        
    # renderer
    ps.register_surface_mesh("my_mesh", V, F)
    ps.register_surface_mesh("my_mesh_simplified", n_V, n_F)

    FaceColor = random_colors(k, FR)
    FaceColor2 = random_colors(n_k, n_FR)

    ps.get_surface_mesh("my_mesh").add_color_quantity(
    "FaceColor",
    FaceColor,
    defined_on="faces")


    ps.get_surface_mesh("my_mesh_simplified").add_color_quantity(
    "FaceColor",
    FaceColor2,
    defined_on="faces")

    ps.show()

    #Matplotlib visualization
    plt.plot(convergence_history_max, label="Max Error")
    plt.plot(convergence_history_mean, label="Mean Error")

    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    export_region_mesh(V, F, FR, output_filepath1)
    export_region_mesh(n_V, n_F, n_FR, output_filepath2)
    np.save(output_filepath3, history)
    np.save(output_filepath4, seeds_all)
    np.save(output_filepath5, order_map)

    return  
"""
flood algorithm (compute R,E):

INPUT: M (V, F), S (seeds), P (proxies), 
OUTPUT: R (regions), E (errors), D (distortion)
local variables: labels, priority queue

0. Push S to priority queue (error, fid, rid) / update labels 

loop start:

1. While priority queue is not empty:
      pop smallest error item

2. If this face is already finalized:
      skip it

3. Otherwise:
      finalize this face with this region (update labels, R, E, D)

4. For each neighboring face:

      if neighbor is already finalized:
          skip

      if this region has already queued this neighbor:
          skip

      if neighbor already has 3 candidates:
          skip

      otherwise:
          compute error (T_i, P_j, W_i: W_i is the local weight)
          push to priority queue/update labels
"""
def flood(mode, V, F, FF, A, W, S, FN, PN, PX, alpha):
    # Labels
    # Label: The first element is the state flag
    # (-1,-1,-1) = unqueued
    # (id1,-1,-1) = queued once by region id1
    # (id1,id2,-1) = queued twice by region id1 and id2
    # (id1,id2,id3) = queued three times by region id1, id2, and id3. should be skipped next time.
    # (-2,winner,...) = poped out and marked by winner region
    # each face can be queued by “label_size” number of regions
    label_size = 3 
    labels = np.full((F.shape[0], label_size), -1, dtype=np.int32)

    # priority queue (error, fid, rid)
    pq = [] 

    # 0.Push the seeds into the priority queue / update labels
    for j, S_j in enumerate(S):

        # error metric: 
        if mode == "L2":
            error = compute_error_L2(V, F, A, W, S_j, PN[j], PX[j])
        elif mode == "L21":
            error = compute_error_L21(A, W, FN, S_j, PN[j])
        elif mode == "H1":
            error = compute_error_H1(V, F, A, W, FN, S_j, PN[j], PX[j], alpha[j])

        # If the error is zero, we can skip it and mark it as finalized
        # otherwise, we push it to the priority queue and mark it as queued
        # they may be finalized by other regions later
        # so the region counts may be less than the number of seeds

        heapq.heappush(pq,(error, S_j, j))
        labels[S_j, 0] = j

    # Loop start
    k = len(S)  # region count
    D = [0.0] * k  # Total distortion of each region
    FE = np.full(len(F), 0.0)  # error of each face
    while pq:

        # 1. Pop the smallest error item
        e, fid, rid_winner = heapq.heappop(pq)
        label = labels[fid]

        # 2. If this face is already finalized, skip it
        if label[0] == -2: 
            continue

        # 3. Otherwise, finalize this face with this region (update labels, R, E, D)
        label[0] = -2
        label[1] = rid_winner
        D[rid_winner] += e
        FE[fid] = e

        # 4. For each neighboring face:
        for fid_neighbor in FF[fid]:
            if fid_neighbor == -1:
                continue

            # if neighbor is already finalized, skip
            if labels[fid_neighbor, 0] == -2:
                continue

            # if this region has already queued this neighbor, skip
            if rid_winner in labels[fid_neighbor]:
                continue

            # if neighbor label is full, skip
            if np.all(labels[fid_neighbor] != -1):
                continue

            # otherwise, compute error (T_i, P_j, W_i: W_i is the local weight) 
            # error metric: change it for different results
            if mode == "L2":
                error = compute_error_L2(V, F, A, W, fid_neighbor, PN[rid_winner], PX[rid_winner])
            elif mode == "L21":
                error = compute_error_L21(A, W, FN, fid_neighbor, PN[rid_winner])
            elif mode == "H1":
                error = compute_error_H1(V, F, A, W, FN, fid_neighbor, PN[rid_winner], PX[rid_winner], alpha[rid_winner])

                # push to priority queue
            heapq.heappush(pq,(error, fid_neighbor, rid_winner))

                # update labels
            for i in range(label_size):
                if labels[fid_neighbor, i] == -1:
                    labels[fid_neighbor, i] = rid_winner    
                    break

    # Face to Region: the winner region id for each face
    FR = labels[:, 1]

    return k, D, FR, FE

#--------------------------------------------------------------------------------------------------------------------------------------






# splitting and merging (proxy fitting is used) ---------------------------------------------------------------------------------------

'''
Fix topology
1. Merge: If the loops that have 1 neighbor
2. Split: If the loops that have 2 neighbors
'''
def topology_handling(mode, VF, k, RV, RA, RR, FR, FM, FE, PN, PX, alpha, g, A, W, FN):

    # If we dynamically modify the regions, The indexing will be changed and no longer valid.
    # But recomputing the topology is expensive, so we propose a simple method to fix the topology without recomputing the topology.
    
    # 1. we record all region ids that need to be merged and all face ids that need to be splitted in the current topology.
    # 2. we merge the regions that need to be merged at once.
    #    now the topology is changed, but we don't care about it anymore.
    # 3. we split all the faces that need to be splitted.
    #    we can savely split the faces because we don't need to know the topology anymore, we just need to know which region each face belongs to.

    # This method avoids ambiguous indexing 
    
    # merging buffer
    chunks = [] # record the region id chunks that need to be merged

    # splitting buffer
    fids = [] # record the face ids that need to be splitted
    splitted = [False] * k # record the regions that have been splitted
    
    for rid in range(k):
        for loop_id in range(len(RR[rid])):

            # merge the loops that have 1 neighbor
            if len(RR[rid][loop_id]) == 1:
                rid_neighbor = RR[rid][loop_id][0]
                if rid_neighbor == -1:
                    continue

                # Check if this region or the neighboring region has already been merged with other regions
                cid_neighbor = -1
                cid_this = -1
                for cid, chunk in enumerate(chunks):
                    if rid_neighbor in chunk:
                        cid_neighbor = cid
                    if rid in chunk:
                        cid_this = cid

                # If no region has been merged with other regions, create a new chunk
                if cid_neighbor == cid_this == -1:
                    chunks.append(set([rid, rid_neighbor]))

                # If only one region has been merged with other regions, add the other region to the same chunk
                elif cid_neighbor == -1:
                    chunks[cid_this].add(rid_neighbor)
                elif cid_this == -1:
                    chunks[cid_neighbor].add(rid)

                # If both regions have been merged with other 2 different region chunks, merge the two chunks
                elif cid_neighbor != cid_this:
                    chunks[cid_neighbor] = chunks[cid_neighbor].union(chunks[cid_this])
                    chunks[cid_this].clear()

            # split the loops that have 2 neighbors 
            if len(RR[rid][loop_id]) == 2:
                # If any neighboring region has been splitted before, we don't need to split it again
                if any(splitted[rid_neighbor] for rid_neighbor in RR[rid][loop_id]):
                    continue

                # If not, find the face incident to 2 anchors with the largest error
                error_max = -1
                fid_max = -1
                rid_max = -1

                for anchor_id in RA[rid][loop_id]:
                    vid = RV[rid][loop_id][anchor_id]
                    for fid in VF[vid]:
                        if fid == -1:
                            continue
                        if FR[fid] not in RR[rid][loop_id]:
                            continue
                        if FE[fid] > error_max:
                            error_max = FE[fid]
                            fid_max = fid
                            rid_max = FR[fid]

                # Split the face with the largest error
                fids.append(fid_max)

                # Mark that regions as splitted
                splitted[rid_max] = True

    # merge all chunks
    k, FR, PX, PN, alpha = merge(mode, chunks, FN, FR, FM, k, PX, PN, alpha, g, A, W)

    # split all faces
    k, FR, PX, PN, alpha = split(mode, k, FR, PX, PN, alpha, g, A, W, FN, FM, fids)

    return k, FR, PX, PN, alpha
'''
Fix local optimum: only when energy distribution is uneven / variance is large)
1. Merge: If two neighboring proxies have smallest error
2. Split: If a proxy has largest error
'''
def teleportation(mode, ratio, V, F, k, FR, FE, D, RR, PX, PN, alpha, g, A, W, FN, FM):

    # Pair to merge
    pairs = {}

    # Bounded heap to find the pair with the smallest merging distortion
    n_diff_heap = []
    c_diff_heap = []
    heap_size = 30
    
    # Region to split
    max_d = -1.0
    max_d_rid = -1

    for rid in range(k):
        # Find the proxy with the largest distortion
        if D[rid] > max_d:
            max_d = D[rid]
            max_d_rid = rid

        # Get the neighboring proxies
        for loop in RR[rid]:
            for rid_neighbor in loop:

                if rid_neighbor == -1:
                    continue

                if (rid_neighbor, rid) in pairs:
                    continue

                # normal difference and centroid difference
                n_diff = np.sum((PN[rid] - PN[rid_neighbor]) ** 2)
                c_diff = np.sum((PX[rid] - PX[rid_neighbor]) * PN[rid]) ** 2 + np.sum((PX[rid] - PX[rid_neighbor]) * PN[rid_neighbor]) ** 2

                pairs[(rid, rid_neighbor)] = (n_diff, c_diff)

                # Find the best top candidates for merging based on normal difference 
                if len(n_diff_heap) < heap_size:
                    heapq.heappush(n_diff_heap, - n_diff)
                elif n_diff > n_diff_heap[0]:
                    heapq.heapreplace(n_diff_heap, - n_diff)

                # Find the best top candidates for merging based on centroid difference
                if len(c_diff_heap) < heap_size:
                    heapq.heappush(c_diff_heap, -c_diff)
                elif c_diff > c_diff_heap[0]:
                    heapq.heapreplace(c_diff_heap, -c_diff)

    # Find cutoff for normal difference and centroid difference
    n_diff_cutoff = - n_diff_heap[0]
    c_diff_cutoff = - c_diff_heap[0]

    # Find the pair with the smallest merging distortion among the candidates
    min_d = float("inf")
    min_d_pair = None
    for pair, (n_diff, c_diff) in pairs.items():
        if n_diff <= n_diff_cutoff and c_diff <= c_diff_cutoff:
            continue
    
        if mode == "L2":
            predicted_d = predict_merge_distortion_L2(V, F, g, A, W, FR, FM, list(pair))
        elif mode == "L21":
            predicted_d = predict_merge_distortion_L21(g, A, W, FN, FR, list(pair))
        elif mode == "H1":
            predicted_d = predict_merge_distortion_H1(V, F, g, A, W, FN, FM, FR, list(pair))

        if predicted_d < max_d * ratio and predicted_d < min_d:
            min_d = predicted_d
            min_d_pair = pair

    if min_d_pair is None:
        print(f"Teleportation skipped: predicted distortion {predicted_d} > max distortion {max_d} * ratio {ratio}")
        return k, FR, PX, PN, alpha, False

    print(f"Teleportation: merging regions ({min_d_pair[0]}, {min_d_pair[1]}) with predicted distortion {min_d} and splitting region {max_d_rid} with max distortion {max_d}")

    # Split at largest error face
    mask = FR == max_d_rid
    fid_list = np.where(mask)[0]
    fid_max_error = fid_list[np.argmax([FE[fid] for fid in fid_list])]

    k, FR, PX, PN, alpha = merge(mode, [set(min_d_pair)], FN, FR, FM, k, PX, PN, alpha, g, A, W)
    k, FR, PX, PN, alpha = split(mode, k, FR, PX, PN, alpha, g, A, W, FN, FM, [fid_max_error])

    return k, FR, PX, PN, alpha, True

def greedy_split(mode, k, FR, FE, PX, PN, alpha, D, g, A, W, FN, FM, threshold=3.0):
    # Find the region with the largest distortion
    max_d = -1.0
    max_d_rid = -1

    for rid in range(k):
        # Find the proxy with the largest distortion
        if D[rid] > max_d:
            max_d = D[rid]
            max_d_rid = rid

    mean_d = np.mean(D)

    if max_d/mean_d < threshold:
        return k, FR, PX, PN, alpha

    # Split at largest error face
    mask = FR == max_d_rid
    fid_list = np.where(mask)[0]
    fid_max_error = fid_list[np.argmax([FE[fid] for fid in fid_list])]
    print(f"Applying greedy split at face {fid_max_error} with error {FE[fid_max_error]} in region {max_d_rid} with distortion {max_d} and mean distortion {mean_d}")

    k, FR, PX, PN, alpha = split(mode, k, FR, PX, PN, alpha, g, A, W, FN, FM, [fid_max_error])
    
    return k, FR, PX, PN, alpha


# splitting
def split(mode, k, FR, PX, PN, alpha, g, A, W, FN, FM, fids):

    for fid in fids:
        old_rid = FR[fid]
       
        # Add a new proxy
        FR[fid] = k
        k += 1
        PX.append(g[fid])
        PN.append(FN[fid])
        alpha.append(np.average(alpha) if mode == "H1" else 0.0)

        # Update the proxy for the old region
        mask = FR == old_rid
        al = 0.0
        if mode == "L2":
            pn, px = vector_compute_proxy_L2(g, A, W, FM, mask)
        elif mode == "L21":
            pn, px = vector_compute_proxy_L21(g, A, W, FN, mask)
        elif mode == "H1":
            pn, px, al = vector_compute_proxy_H1(g, A, W, FN, FM, mask)

        PN[old_rid] = pn
        PX[old_rid] = px
        alpha[old_rid] = al

    return k, FR, PX, PN, alpha
# merging (chunks: list of sets)
def merge(mode, rid_chunks, FN, FR, FM, k, PX, PN, alpha, g, A, W):

    r_count = k
    id_deleted = []
    for chunk in rid_chunks:
        
        if not chunk:
            continue

        # compute new proxy for the merged region
        rid_list = list(chunk)
        new_al = 0.0
        mask = np.isin(FR, rid_list)
        if mode == "L2":
            new_pn, new_px = vector_compute_proxy_L2(g, A, W, FM, mask)
        elif mode == "L21":
            new_pn, new_px = vector_compute_proxy_L21(g, A, W, FN, mask)
        elif mode == "H1":
            new_pn, new_px, new_al= vector_compute_proxy_H1(g, A, W, FN, FM, mask)

        # mark the old regions as deleted
        id_deleted.extend(rid_list)
        for rid in rid_list:
            PN[rid] = None
            PX[rid] = None
            alpha[rid] = None

        # update FR, PX, PN, alpha
        FR[mask] = r_count
        PX.append(new_px)
        PN.append(new_pn)   
        alpha.append(new_al)
        r_count += 1

    # clean up regions
    new_PX = []
    new_PN = []
    new_alpha = []
    new_FR = np.full(len(FR), -1, dtype=np.int32)

    new_rid = 0
    for i in range(r_count):

        if i in id_deleted:
            continue
        else:
            new_PX.append(PX[i])
            new_PN.append(PN[i])
            new_alpha.append(alpha[i])
            new_FR[FR == i] = new_rid
            new_rid += 1

    new_k = new_rid

    return new_k, new_FR, new_PX, new_PN, new_alpha

#--------------------------------------------------------------------------------------------------------------------------------------






# Different metrics -------------------------------------------------------------------------------------------------------------------

# L2 metric
def compute_error_L2(V, F, A, W, fid, PN_j, PX_j): # W_i is the weight for fid
    AW_i = A[fid] * W[fid]
    v1 = V[F[fid, 0]]
    v2 = V[F[fid, 1]]
    v3 = V[F[fid, 2]]
    d1 = np.dot(v1 - PX_j, PN_j)
    d2 = np.dot(v2 - PX_j, PN_j)
    d3 = np.dot(v3 - PX_j, PN_j)
    E_i = AW_i * (d1**2 + d2**2 + d3**2 + d1*d2 + d2*d3 + d1*d3) / 6.0
    
    return E_i
def vector_compute_proxy_L2(g, A, W, FM, region_mask):

    g_i = g[region_mask]
    A_i = A[region_mask]
    W_i = W[region_mask]
    FM_i = FM[region_mask]
    AW_i = A_i * W_i
    RA_j = AW_i.sum()

    if RA_j == 0:
        return None, None
    
    PX_j = (AW_i[:, np.newaxis] * g_i).sum(axis=0) / RA_j
    
    M_cov = (AW_i[:, np.newaxis, np.newaxis] * FM_i).sum(axis=0) - RA_j * (PX_j[:, None] @ PX_j[None, :])

    # Compute the eigenvector corresponding to the smallest eigenvalue
    eigvals, eigvecs = np.linalg.eigh(M_cov)
    id_min = np.argmin(eigvals)
    PN_j = eigvecs[:, id_min] # [:,_] means all rows
    PN_j /= np.linalg.norm(PN_j) # normalize the vector

    return PN_j, PX_j
def compute_best_triangle_L2(V, F, A, W, FR, PN, PX, rid):
    mask = FR == rid
    region_fids = np.flatnonzero(mask)

    AW_j = A[mask] * W[mask]

    V1 = V[F[mask, 0]]
    V2 = V[F[mask, 1]]
    V3 = V[F[mask, 2]]

    d1 = (V1 - PX[rid]) @ PN[rid]
    d2 = (V2 - PX[rid]) @ PN[rid]
    d3 = (V3 - PX[rid]) @ PN[rid]

    E = AW_j * ( d1**2 + d2**2 + d3**2 + d1*d2 + d2*d3 + d1*d3) / 6.0

    fid_min = region_fids[np.argmin(E)]

    return fid_min
def predict_merge_distortion_L2(V, F, g, A, W, FR, FM, rids): 
    mask = np.isin(FR, rids)
    pn, px = vector_compute_proxy_L2(g, A, W, FM, mask)

    AW_j = A[mask] * W[mask]

    V1 = V[F[mask, 0]]
    V2 = V[F[mask, 1]]
    V3 = V[F[mask, 2]]

    d1 = (V1 - px) @ pn
    d2 = (V2 - px) @ pn
    d3 = (V3 - px) @ pn

    E = AW_j * ( d1**2 + d2**2 + d3**2 + d1*d2 + d2*d3 + d1*d3) / 6.0
    D = E.sum()
    return D

# L2,1 metric
def compute_error_L21(A, W, FN, fid, PN_j): # W_i is the weight for fid
    AW_i = A[fid] * W[fid]
    n_i = FN[fid]
    E_i = AW_i * np.sum((n_i - PN_j) ** 2)
    
    return E_i
def vector_compute_proxy_L21(g, A, W, FN, region_mask):
    g_i = g[region_mask]
    A_i = A[region_mask]
    W_i = W[region_mask]
    AW_i = A_i * W_i
    RA_j = AW_i.sum()

    weighted_n = (AW_i[:, None] * FN[region_mask]).sum(axis=0)
    norm = np.linalg.norm(weighted_n) # normalize the vector

    if norm == 0 or RA_j == 0:
        return None, None

    PN_j = weighted_n / norm

    PX_j = (AW_i[:, None] * g_i).sum(axis=0) / RA_j

    return PN_j, PX_j
def compute_best_triangle_L21(A, W, FN, FR, PN, rid):
    mask = FR == rid
    region_fids = np.flatnonzero(mask)

    AW_j = A[mask] * W[mask]
    n_j = FN[mask]

    E = AW_j * np.sum((n_j - PN[rid]) ** 2, axis=1)

    fid_min = region_fids[np.argmin(E)]

    return fid_min
def predict_merge_distortion_L21(g, A, W, FN, FR, rids):
    mask = np.isin(FR, rids)
    pn, _ = vector_compute_proxy_L21(g, A, W, FN, mask)

    AW_j = A[mask] * W[mask]
    n_j = FN[mask]

    E = AW_j * np.sum((n_j - pn) ** 2, axis=1)
    D = E.sum()
    return D

# Sobolev H1 metric
gamma = 0.45 # (0 < gamma < infinity)
def compute_error_H1(V, F, A, W, FN, fid, PN_j, PX_j, alpha_j): # W_i is the weight for fid
    AW_i = A[fid] * W[fid]
    
    v1 = V[F[fid, 0]]
    v2 = V[F[fid, 1]]
    v3 = V[F[fid, 2]]

    d1 = np.dot(v1 - PX_j, PN_j)
    d2 = np.dot(v2 - PX_j, PN_j)
    d3 = np.dot(v3 - PX_j, PN_j)

    d2 = (d1**2 + d2**2 + d3**2 + d1*d2 + d2*d3 + d1*d3) / 6.0
    E_L2 = AW_i * d2

    n_i = FN[fid]
    E_L21 = AW_i * np.sum((n_i - PN_j) ** 2)

    if alpha_j == 0.0 or alpha_j is None:
        alpha_j = gamma**2

    E_i = E_L2 + alpha_j * E_L21

    return E_i
# root solver needed. adaptive alpha is selected.
def vector_compute_proxy_H1(g, A, W, FN, FM, region_mask):
    g_i = g[region_mask]
    A_i = A[region_mask]
    W_i = W[region_mask]
    N_i = FN[region_mask]
    FM_i = FM[region_mask]
    AW_i = A_i * W_i
    m_j = (AW_i[:, np.newaxis] * N_i).sum(axis=0)
    RA_j = AW_i.sum()

    if RA_j == 0:
        return None, None
    
    PX_j = (AW_i[:, np.newaxis] * g_i).sum(axis=0) / RA_j
    
    M_cov = (AW_i[:, np.newaxis, np.newaxis] * FM_i).sum(axis=0) - RA_j * (PX_j[:, None] @ PX_j[None, :])

    # Compute the eigenvector corresponding to the smallest eigenvalue
    lam, U = np.linalg.eigh(M_cov)

    a = U.T @ m_j

    eps = 1e-12

    alpha_j = gamma**2 #* (lam[2]+lam[1])/(RA_j*2)

    def f(rho):
        return alpha_j**2 * np.sum(a**2 / (lam - rho)**2) - 1.0

    upper = lam[0] - eps
    lower = lam[0] - 1.0

    while f(upper) < 0: # exponential search for an upper bound
        eps /= 2.0
        upper += eps

    while f(lower) > 0: # exponential search for a lower bound
        lower -= 2.0 * (upper - lower)

    rho = brentq(f, lower, upper)

    b = alpha_j * a / (lam - rho)

    PN_j = U @ b
    PN_j /= np.linalg.norm(PN_j)

    return PN_j, PX_j, alpha_j
def compute_best_triangle_H1(V, F, A, W, FN, FR, PN, PX, alpha, rid):
    mask = FR == rid
    region_fids = np.flatnonzero(mask)

    AW_i = A[mask] * W[mask] 
    n_j = FN[mask]
    alpha_j = alpha[rid]
    

    V1 = V[F[mask, 0]]
    V2 = V[F[mask, 1]]
    V3 = V[F[mask, 2]]

    d1 = (V1 - PX[rid]) @ PN[rid]
    d2 = (V2 - PX[rid]) @ PN[rid]
    d3 = (V3 - PX[rid]) @ PN[rid]

    d2 = (d1**2 + d2**2 + d3**2 + d1*d2 + d2*d3 + d1*d3) / 6.0
    E_L2 = AW_i * d2
    E_L21 = AW_i * np.sum((n_j - PN[rid]) ** 2, axis=1)

    if alpha_j == 0.0 or alpha_j is None:
        alpha_j = gamma**2

    E = E_L2 + alpha_j * E_L21

    fid_min = region_fids[np.argmin(E)]

    return fid_min
def predict_merge_distortion_H1(V, F, g, A, W, FN, FM, FR, rids):
    mask = np.isin(FR, rids)
    pn, px, al = vector_compute_proxy_H1(g, A, W, FN, FM, mask)

    AW_i = A[mask] * W[mask]

    V1 = V[F[mask, 0]]
    V2 = V[F[mask, 1]]
    V3 = V[F[mask, 2]]

    d1 = (V1 - px) @ pn
    d2 = (V2 - px) @ pn
    d3 = (V3 - px) @ pn

    d2 = (d1**2 + d2**2 + d3**2 + d1*d2 + d2*d3 + d1*d3) / 6.0
    E_L2 = AW_i * d2
    E_L21 = AW_i * np.sum((FN[mask] - pn) ** 2, axis=1)

    E = E_L2 + al * E_L21

    D = E.sum()
    return D

#--------------------------------------------------------------------------------------------------------------------------------------






# Meshing -----------------------------------------------------------------------------------------------------------------------------
# Construct discrete delauney triangulation (CDT) for each region
def simplify_mesh(V, F, FR, FF, VV, VF, PX, PN, collapse_threshold=np.float64(0.0)):
# 1. Get all the boundary vertices, edges, and anchors
   
    vv_boundary_adj = [[] for _ in range(len(V))]
    edges_boundary = set()
    for fid in range(F.shape[0]):
        rid = FR[fid]
        for eid, fid_neighbor in enumerate(FF[fid]):

            # Get region ID at this neighbor face
            rid_neighbor = FR[fid_neighbor] if fid_neighbor != -1 else -1

            if rid_neighbor == rid:
                continue

            # Oriented mesh has half edge duality, so we only need to mark the start vertex of each boundary edge
            vid_start = F[fid, eid]
            vid_end   = F[fid, (eid + 1) % 3]
            edges_boundary.add((min(vid_start, vid_end), max(vid_start, vid_end)))

            vv_boundary_adj[vid_start].append(vid_end)
            if rid_neighbor == -1:
                vv_boundary_adj[vid_end].append(vid_start)

    # Find all vertices that are connected to more than 2 vertices
    # These vertices are the anchors that we will use to flood the other vertices
    # Compute the projection of these anchor vertices to the proxy planes of the regions they belong to, and update their positions accordingly.
    # Interestingly, when the mesh is too small or big, float32 may lead to precision issues that stops the queue poping out the correct result in eager dijkstra, but the lazy dijkstra is more robust. 
    seeds = np.full(len(V), -1, dtype=np.int32)
    dists = np.full(len(V), np.inf, dtype=np.float64)
    new_V = V.copy()
    for vid in range(len(V)):
        if len(vv_boundary_adj[vid]) > 2:
            seeds[vid] = vid
            dists[vid] = 0.0

            rid_set = set()
            v_proj = np.zeros(3, dtype=np.float64)
            for fid in VF[vid]:
                if fid == -1:
                    continue
                if FR[fid] == -1:
                    continue
                if FR[fid] in rid_set:
                    continue

                # project vertex to the proxy plane of the region
                p_normal = PN[FR[fid]]
                p_point = PX[FR[fid]]
                p_proj = V[vid] - np.dot(V[vid] - p_point, p_normal) * p_normal
                v_proj += p_proj
                rid_set.add(FR[fid])

            v_proj /= len(rid_set)
            new_V[vid] = v_proj


# 2. flood boundary vertices from anchors (dijkstra)
    seeds_constraint, dists_constraint, boundary_order_map = dijkstra_voronoi(V, vv_boundary_adj, seeds, dists)

# 3. Selective: If 2 or more anchors are too close, we can merge them into one anchor by assigning the same anchor id to them. 
    seed_merge_chunks = []
    for edge in edges_boundary:
        vid_start, vid_end = edge
        seed_start = seeds_constraint[vid_start]
        seed_end = seeds_constraint[vid_end]

        if seed_start != -1 and seed_end != -1 and seed_start != seed_end: # If both vertices are anchors and they are different anchors
            dist_start = dists_constraint[vid_start]
            dist_end = dists_constraint[vid_end]
            dist_sum = dist_start + dist_end

            # Add to the merge chunks if the distance is smaller than the threshold
            if dist_sum < collapse_threshold:
                cid_start = -1
                cid_end = -1
                for cid, chunk in enumerate(seed_merge_chunks):
                    if seed_start in chunk:
                        cid_start = cid
                    if seed_end in chunk:
                        cid_end = cid

                # If both seeds are not in any chunk, create a new chunk
                if cid_start == -1 and cid_end == -1:
                    seed_merge_chunks.append(set([seed_start, seed_end]))

                # If only one seed is in a chunk, add the other seed to the same chunk
                elif cid_end == -1:
                    seed_merge_chunks[cid_start].add(seed_end)
                elif cid_start == -1:
                    seed_merge_chunks[cid_end].add(seed_start)

                # If both seeds are in different chunks, merge the two chunks
                elif cid_start != cid_end:
                    seed_merge_chunks[cid_start] = seed_merge_chunks[cid_start].union(seed_merge_chunks[cid_end])
                    seed_merge_chunks[cid_end].clear()

    # Get chunk heads (the smallest vid seed in each chunk. If a chunk is empty, the head is -1)
    # Update the seed(vertex)'s position to the average position of all seeds in the chunk
    chunk_heads = []
    for chunk in seed_merge_chunks:
        if chunk:
            head = min(chunk)

            # Update the position of the head vertex to the average position of all vertices in the chunk
            vids = np.array([int(v) for v in chunk], dtype=np.int64)
            new_V[head] = np.mean(new_V[vids], axis=0)

            chunk_heads.append(head)
            print(f"Merging seeds {[int(x) for x in chunk]} into {int(head)}")

        else:
            chunk_heads.append(-1)

    # Replace the seeds
    for sid, seed in enumerate(seeds_constraint):
        if seed == -1:
            continue
        for cid, chunk in enumerate(seed_merge_chunks):
            if seed in chunk:
                seeds_constraint[sid] = chunk_heads[cid]  # Assign the smallest seed id in the chunk to all seeds in the chunk
                break

    # debugger: display merged seed vertices
    for cid, chunk in enumerate(seed_merge_chunks):
        if len(chunk) < 2:
            continue
        vids = np.array([int(v) for v in chunk], dtype=np.int64)
        color = np.random.rand(3).tolist()
        ps.register_point_cloud(
            f"merged_chunk_{cid}",
            V[vids],
            radius=0.004,
            color=color
        )

# 4. flood inner vertices from anchors (dijkstra)
    seeds_boundary = seeds_constraint.copy()
    seeds_final, _, inner_order_map = dijkstra_voronoi(V, VV, seeds_constraint, dists_constraint)
    seeds_all = seeds_final.copy()
    order_map = np.where(inner_order_map != -1, inner_order_map, boundary_order_map)

# 5. trianglulate each region by the the faces with 3 vertices assigned to 3 different anchors
    
    # There is a potential problem that this method may generate non-manifold edges when more than there anchor seeds have similar distances to a vertex. 
    # When differnet cadidates have the same distance to a vertex, the eager dijkstra will assign the vertex to the first candidate that is popped from the priority queue.
    # It may result in a vertex being labeled by an anchor seed (for example A) that not existing in its neiboring vertices (neighbors are assigned to B, C, D...)
    # it will generate an extra face of (B, C, D)

    # It barely happens, but for the robustness, we should test if there is an isolated labeling of a vertex, and relabel it to any anchor seed in its neighboring vertices.
    for vid in range(len(V)):
        isolated = True
        for vid_neighbor in VV[vid]:
            if seeds_final[vid_neighbor] == seeds_final[vid]:
                isolated = False
                break
        if isolated:
            seeds_final[vid] = seeds_final[VV[vid][0]] # relabel it to any anchor seed in its neighboring vertices
            print(f"Warning: Vertex {vid} is isolated when triangulating. Relabeling to {seeds_final[vid]}.")
    
    # Delaunay triangulation: The faces where 3 anchor seeds define the triangles because of the duality between the Voronoi diagram and the Delaunay triangulation.
    # But sometimes, the face may drops into a region that does not belong to any of the 3 anchor seeds
    # Usually its fine, but if the mis-assigned region only have 2 neighboring regions, then it will become a topology error later.
    new_F = []
    new_FR = []
    for fid in range(F.shape[0]):
        a1 = seeds_final[F[fid, 0]]
        a2 = seeds_final[F[fid, 1]]    
        a3 = seeds_final[F[fid, 2]]
        rid = FR[fid]

        # Some vertex was not assigned to an anchor
        if a1 == -1 or a2 == -1 or a3 == -1:
            print(f"Warning: Face {fid} has unassigned vertex. Skipping.")
            continue

        # triangle with 2 or 3 vertices assigned to the same anchor, skip it
        if len({a1, a2, a3}) != 3:
            continue

        new_F.append([a1, a2, a3])
        new_FR.append(rid)

    return new_V, np.array(new_F, dtype=np.int32), np.array(new_FR, dtype=np.int32), seeds_all, seeds_boundary, order_map          
def dijkstra_voronoi(V, vv_adj, seeds_constraint, min_dist_constraint):

    min_seeds = seeds_constraint
    min_dists = min_dist_constraint

    constraint_mask = min_seeds != -1

    order_ptr = len(min_seeds[constraint_mask])
    order_map = np.full(len(V), -1, dtype=np.int32)
    
    pq = []
    for vid, seed in enumerate(min_seeds):
        if seed == -1:
            continue
        dist = min_dists[vid]
        heapq.heappush(pq, (dist, vid, seed))  # (distance, vertex_id, seed_id)

    while pq:
        dist, vid, seed_vid = heapq.heappop(pq)

        # Skip if the current distance is greater than the minimum distance for this vertex
        if dist > min_dists[vid]:
            continue  

        for vid_neighbor in vv_adj[vid]:
            # If the neighbor is constrained, skip it
            if constraint_mask[vid_neighbor]:
                continue

            new_dist = dist + np.linalg.norm(V[vid] - V[vid_neighbor])  # Compute the distance to the neighbor

            if new_dist < min_dists[vid_neighbor]:  # Only consider this neighbor if it offers a shorter path
                # Eager update: directly update the minimum seed and distance when pushing to the priority queue
                # For lazy update, you can push to the priority queue without updating min_seeds and min_dists, and only update them when popping from the queue.
                
                # Eager update
                min_seeds[vid_neighbor] = seed_vid  # Assign the seed ID to the vertex
                min_dists[vid_neighbor] = new_dist  # Update the minimum distance for the neighbor

                order_map[vid_neighbor] = order_ptr
                order_ptr += 1

                heapq.heappush(pq, (new_dist, vid_neighbor, seed_vid))  # Push the neighbor onto the priority queue

    return min_seeds, min_dists, order_map    
"""
    Cotangent-weighted Taubin mesh smoothing.
    V : (n, 3) float
        Vertex positions.
    F : (m, 3) int
        Triangle indices.
    iterations : int
        Number of lambda/mu pairs.
    lam : float
        Positive Taubin lambda parameter.
    mu : float
        Negative Taubin mu parameter.
    """
def cotan_taubin_smooth(V, F, iterations=100, lam=0.6, mu=-0.23):
    V = np.asarray(V, dtype=np.float64).copy()
    F = np.asarray(F, dtype=np.int64)

    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError("V must have shape (n, 3).")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("F must have shape (m, 3).")

    # Cotangent Laplacian.
    # Libigl convention: L is negative semidefinite.
    L = igl.cotmatrix(V, F)

    # Lumped mass matrix.
    M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)

    mass = M.diagonal()
    if not np.all(np.isfinite(mass)):
        raise ValueError("Mass matrix contains NaN/Inf.")

    if np.any(mass <= 0):
        raise ValueError("Mesh contains non-positive Voronoi masses.")

    # M^{-1}
    inv_mass = 1.0 / mass

    # Normalize the cotangent Laplacian spectrally.
    # B = -M^{-1/2} L M^{-1/2}
    # B is symmetric positive semidefinite.
    sqrt_inv_mass = 1.0 / np.sqrt(mass)
    D = sp.diags(sqrt_inv_mass)

    B = D @ (-L) @ D
    B = (B + B.T) * 0.5

    # Largest eigenvalue of B.
    lambda_max = eigsh(B, k=1, which="LA", return_eigenvectors=False)[0]
    lambda_max = float(lambda_max)
    if not np.isfinite(lambda_max) or lambda_max <= 0:
        return V

    # Normalized cotangent Laplace-Beltrami operator:
    # T = M^{-1} L / lambda_max
    # Its eigenvalues are approximately in [-1, 0].
    def laplace(X):
        return (inv_mass[:, None] * (L @ X)) / lambda_max

    # Taubin filter:
    #   lambda pass:
    #       V <- V + lambda * T V
    #   mu pass:
    #       V <- V + mu * T V
    # Since lambda > 0 and eigenvalues(T) <= 0,
    # the first pass damps high frequencies.
    # Since mu < 0, the second pass compensates low-frequency shrinkage.

    for _ in range(iterations):

        # Lambda pass
        V = V + lam * laplace(V)

        # Mu pass
        V = V + mu * laplace(V)

        if not np.all(np.isfinite(V)):
            raise ValueError(
                "Taubin smoothing produced NaN/Inf vertices."
            )

    return V
def face_normal_smooth(FN, FF, A, iterations=10, cos_threshold=0.8):
    FN_smooth = np.asarray(FN, dtype=np.float64).copy()
    A = np.asarray(A, dtype=np.float64)

    for _ in range(iterations):
        new_FN = FN_smooth * A[:, None]
        W = A.copy()

        for i in range(3):
            fid_neighbors = FF[:, i]

            valid = fid_neighbors != -1

            ids = fid_neighbors[valid]

            # Cosine between current face normal and neighbor normal
            cos = np.sum(FN_smooth[valid] * FN_smooth[ids], axis=1)

            accepted = cos > cos_threshold

            valid_ids = np.where(valid)[0][accepted]
            neighbor_ids = ids[accepted]

            weights = A[neighbor_ids]

            new_FN[valid_ids] += (weights[:, None] * FN_smooth[neighbor_ids])

            W[valid_ids] += weights

        FN_smooth = new_FN / W[:, None]

        norms = np.linalg.norm(FN_smooth, axis=1)

        valid_norm = norms > 1e-15
        FN_smooth[valid_norm] /= norms[valid_norm, None]

    return FN_smooth

#--------------------------------------------------------------------------------------------------------------------------------------






# Planar Polygon ----------------------------------------------------------------------------------------------------------------------
def compute_polygons(PX, PN, RR, V, RV, RA):
    # Compute plane equations
    n_PX = np.array(PX)
    n_PN = np.array(PN)

    D = np.sum(n_PN * n_PX, axis=1)
    Polygons = []

    for rid in range(len(RR)):
        n1 = n_PN[rid]
        d1 = D[rid]

        polygon = []

        for loop_id in range(len(RR[rid])):
            rids_count = len(RR[rid][loop_id])

            polyline = []

            for i in range(rids_count):
                rid_start = RR[rid][loop_id][i-1]
                rid_end = RR[rid][loop_id][i]
                point = V[RV[rid][loop_id][RA[rid][loop_id][i]]]

                if rid_start == -1 and rid_end == -1:
                    continue

                # Compute the intersection point of the 3 planes (rid, rid_start, rid_end)
                n2 = n_PN[rid_start]
                d2 = D[rid_start]
                n3 = n_PN[rid_end]
                d3 = D[rid_end]

                if rid_start == -1 or rid_end == -1:
                    # project the point onto the intersection line of the two planes (rid, rid_start) or (rid, rid_end)
                    if rid_start != -1:
                        point = project_point_to_plane_intersection(point, n1, d1, n2, d2)
                    else:
                        point = project_point_to_plane_intersection(point, n1, d1, n3, d3)

                    polyline.append(point)
                    continue

                A = np.array([n1, n2, n3])
                b = np.array([d1, d2, d3])

                try:
                    intersection_point = np.linalg.solve(A, b)
                    if np.sum((intersection_point-point)**2) > 100:
                        intersection_point = point

                        print(f"Warning: Intersection point for rid {rid}, loop {loop_id}, edge ({rid_start}, {rid_end}) is far from previous point.")
                except np.linalg.LinAlgError:
                    print(f"Cannot compute intersection point for rid {rid}, loop {loop_id}, edge ({rid_start}, {rid_end})")
                    continue

                polyline.append(intersection_point)

            polygon.append(polyline)

        Polygons.append(polygon)

    return Polygons
def project_point_to_plane_intersection(q, n1, d1, n2, d2):

    v = np.cross(n1, n2)
    vv = np.dot(v, v)

    if vv < 1e-14:
        # No unique intersection line.
        # Fall back to projection onto plane 1.
        return q - (np.dot(n1, q) - d1) / np.dot(n1, n1) * n1

    # Find a point p on the intersection line.
    A = np.vstack([n1, n2, v])
    b = np.array([d1, d2, 0.0], dtype=np.float64)

    p = np.linalg.solve(A, b)

    # Closest point on the intersection line to q.
    t = np.dot(q - p, v) / vv
    q_projected = p + t * v

    return q_projected
def plane_fit_anchors(V, k, RR, RV, RA, PX, PN):

    errs = [0.0] * k
    normals = PN.copy()
    centers = PX.copy()
    for rid in range(k):
        anchor_points = []
        for loop_id in range(len(RA[rid])):
            for anchor_id in enumerate(RA[rid][loop_id]):
                vid = RV[rid][loop_id][anchor_id]
                v = V[vid]
                anchor_points.append(v)

        if len(anchor_points) < 3:
            errs[rid] = None
            normals[rid] = None
            centers[rid] = None
            continue

        anchor_points = np.array(anchor_points)

        # centralize data
        c = np.mean(anchor_points, axis = 0)
        #c = centers[rid]
        
        centered_points = anchor_points - c

        # covariance
        M_cov = np.dot(centered_points.T, centered_points)
        lam, U = np.linalg.eigh(M_cov)
        u = U[:, 0]

        u/= np.linalg.norm(u) # normalize the vector

        # new plane
        error = lam[0] / len(anchor_points)
        normal = u
        center = c

        errs[rid] = error
        normals[rid] = normal
        centers[rid] = center






    # Debug visualization
    centers_np = np.asarray(centers, dtype=np.float64)
    normals_np = np.asarray(normals, dtype=np.float64)

    # Ignore invalid regions
    valid = np.array([
        c is not None and n is not None
        for c, n in zip(centers, normals)
    ])

    centers_np = centers_np[valid]
    normals_np = normals_np[valid]

    # Normal arrows
    scale = 1

    points = np.empty((2 * len(centers_np), 3), dtype=np.float64)
    points[0::2] = centers_np
    points[1::2] = centers_np + scale * normals_np

    edges = np.column_stack([
        np.arange(0, 2 * len(centers_np), 2),
        np.arange(1, 2 * len(centers_np), 2)
    ]).astype(np.int32)

    ps.register_curve_network(
        "plane_normals",
        points,
        edges,
        radius=0.001
    )

    return errs, normals, centers
def triangulate_polygons_cdt(polygons):

    import triangle as tr

    all_vertices = []
    all_faces = []
    offset = 0

    for polygon in polygons:

        if not polygon or len(polygon[0]) < 3:
            continue

        # Convert first loop to numpy
        outer = np.asarray(polygon[0], dtype=float)

        # Remove duplicated closing point
        if len(outer) > 2 and np.allclose(outer[0], outer[-1]):
            outer = outer[:-1]

        if len(outer) < 3:
            continue

        # Fit plane
        p0 = np.mean(outer, axis=0)
        X = outer - p0
        C = X.T @ X
        _, eigenvectors = np.linalg.eigh(C)
        normal = eigenvectors[:, 0]
        normal /= np.linalg.norm(normal)


        normal_length = np.linalg.norm(normal)

        if normal_length < 1e-12:
            continue

        normal /= normal_length

        # Stable reference direction
        if abs(normal[0]) < 0.9:
            ref = np.array([1., 0., 0.])
        else:
            ref = np.array([0., 1., 0.])

        u = np.cross(normal, ref)
        u /= np.linalg.norm(u)

        v = np.cross(normal, u)

        # 3D -> 2D
        loops_2d = []

        for loop in polygon:

            pts = np.asarray(loop, dtype=float)

            if len(pts) > 2 and np.allclose(pts[0], pts[-1]):
                pts = pts[:-1]

            if len(pts) < 3:
                continue

            d = pts - p0
            xy = np.column_stack([ d @ u, d @ v])
            loops_2d.append(xy)

        if not loops_2d:
            continue

        # Build vertices + constrained segments
        vertices = []
        segments = []
        vertex_map = {}

        def get_id(p):

            key = tuple(np.round(p, 10))

            if key not in vertex_map:
                vertex_map[key] = len(vertices)
                vertices.append(p)

            return vertex_map[key]

        for loop in loops_2d:
            ids = [get_id(p) for p in loop]
            for i in range(len(ids)):
                a = ids[i]
                b = ids[(i + 1) % len(ids)]

                if a != b:
                    segments.append([a, b])

        vertices = np.asarray(vertices, dtype=float)
        segments = np.asarray(segments, dtype=np.int32)

        # Hole points
        # IMPORTANT:
        # centroid/mean is not guaranteed to be inside a
        # concave hole.
        # For now use the mean, but verify it.
        holes = []

        for loop in loops_2d[1:]:

            # polygon centroid
            x = loop[:, 0]
            y = loop[:, 1]

            cross = (x * np.roll(y, -1) - np.roll(x, -1) * y)
            area = 0.5 * np.sum(cross)

            if abs(area) < 1e-12:
                continue

            cx = np.sum((x + np.roll(x, -1)) * cross) / (6.0 * area)
            cy = np.sum((y + np.roll(y, -1)) * cross) / (6.0 * area)
            holes.append([cx, cy])

        # Triangle input

        data = {"vertices": vertices, "segments": segments}

        if holes:
            data["holes"] = np.asarray(holes,dtype=float)

        # Constrained Delaunay triangulation
        result = tr.triangulate(data,"p")

        if "triangles" not in result:
            continue

        verts2d = result["vertices"]
        faces = result["triangles"]

        # 2D -> 3D
        verts3d = (p0 + verts2d[:, 0, None] * u + verts2d[:, 1, None] * v)

        # Accumulate
        all_vertices.extend(verts3d)
        all_faces.extend(faces + offset)
        offset += len(verts3d)


    return (np.asarray(all_vertices, dtype=float),np.asarray(all_faces, dtype=np.int32))

#--------------------------------------------------------------------------------------------------------------------------------------

