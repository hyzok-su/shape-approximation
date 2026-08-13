import heapq
from xml.parsers.expat import errors
import numpy as np
import igl
import polyscope as ps

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
    colors = np.random.rand(k, 3)
    face_colors = np.zeros((len(FR), 3))
    for fid in range(len(FR)):
        rid = FR[fid]
        face_colors[fid] = colors[rid]

    return face_colors




# Preprocessing -----------------------------------------------------------------------------------------------------------------------

def load_mesh(path):
    V, F = igl.read_triangle_mesh(path)
    V, F, _, _ = igl.remove_unreferenced(V, F)
    if F.shape[0] == 0:
        raise ValueError(f"No faces found when reading '{path}'")
    F_oriented, _ = igl.bfs_orient(F)
    return V, F_oriented

# Compute per face g, area, normal
def compute_face_properties(V, F):
    g = np.mean(V[F], axis=1)
    A = igl.doublearea(V, F) / 2.0
    FN = igl.per_face_normals(V, F) # Can be smoothed if needed

    return g, A, FN

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
                    '''
                    # debugger
                    if edge not in pinched_vid_next_map:
                        debug_render_region_pinches(
                        rid=rid,
                        F=F,
                        V=V,
                        FR=FR,
                        RP=RP,
                        RP_next_maps=RP_next_maps,
                        failing_edge=edge,
                    )
                    '''
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

                    '''
                    # debugger
                    edge = (vid_this, vid_next)
                    print(len(RP[rid])==0)
                    debug_render_region_pinches(
                        rid=rid,
                        F=F,
                        V=V,
                        FR=FR,
                        RP=RP,
                        RP_next_maps=RP_next_maps,
                        failing_edge=edge
                    )
                    '''
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

# Get the best seed triangle at each region (proxy)
def compute_best_triangle(FR, FE, rid):
    mask = FR == rid
    fids = np.where(mask)[0]
    fid_min = fids[np.argmin(FE[mask])]

    return fid_min

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


    



# VSA (plane proxy) ===================================================================================================================

"""
VSA: Variational Shape Approximation
1. Flood: S, P ----> R, E
2. Proxy Fitting + Best Triangle Selection: R, E ----> S, P
3. Proxy insertion + deletion: S, P ----> S, P
repeat 1-3 until convergence
"""
def VSA(mode, path, init_seeds):

    # Load mesh
    V, F = load_mesh(path)

    ps.init()
    ps.set_ground_plane_mode("none")

    # Face Properties
    g, A, FN = compute_face_properties(V, F)
    W = np.ones(F.shape[0], dtype=float) # list of face artificial weights

    # Proxies
    S = list(init_seeds) # list of seed fids for each proxy
    k = len(init_seeds) # number of proxies
    PN = list(FN[S]) # list of proxy normals
    PX = list(g[S]) # list of proxy centroids

    # Adjacency Lists
    # Face-face adjacency
    FF, _ = igl.triangle_triangle_adjacency(F)
    # Vertice-face adjacency
    VF, VV, _ = vertex_face_adjacency(V, F, FF)

    iterations = 100
    for iteration in range(iterations):
        # Compute regions
        k, D, FR, FE = flood(mode, V, F, FF, A, W, S, FN, PN, PX)
        k, FR, D, _ = rebuild_region(FR, FF, FE)
        RV, RA, RR, topology_errors = compute_region_topology(F, k, FR, FF, VF, VV)

        #show_region_loop_debug(V, F, FR, VF, RV, RA, rid=20, loop_id=0)
        
        # Update Proxies
        for rid in range(k):
            mask = FR == rid
            if mode == "L2":
                pn, px = vector_compute_proxy_L2(V, F, g, A, W, mask)
            elif mode == "L21":
                pn, px = vector_compute_proxy_L21(g, A, W, FN, mask)

            PN[rid] = pn
            PX[rid] = px

        # merge and split regions
        if iteration != iterations-1:
            if iteration < iterations*3/4 and iteration % 10 == 0:  
                k, FR, PX, PN = update_proxies_by_teleportation(mode, k, FR, FE, D, RR, PX, PN, V, F, g, A, W, FN)
            
            
            elif topology_errors > 0 and iteration > iterations*1/2 and iteration % 5 == 0:
                # This method may try to split a region only has one face, which may result in a proxy with no faces assigned to it.
                # So we need to clean up the lists later.
                k, FR, PX, PN = update_proxies_by_topology(mode, VF, k, RV, RA, RR, FR, FE, PN, PX, V, F, g, A, W, FN)
        

        # Clean up: remove proxies with no faces assigned to them
        new_S = []
        new_PN = []
        new_PX = []
        for rid in range(k):
            if np.any(FR == rid):
                new_S.append(compute_best_triangle(FR, FE, rid))
                new_PN.append(PN[rid])
                new_PX.append(PX[rid])

        S = new_S
        PN = new_PN
        PX = new_PX

        print(f"Iteration {iteration}: {k} proxies, max error: {max(D)}, mean error: {sum(D)/k if k > 0 else 0}, topology errors: {topology_errors}")

    
    n_F, n_FR = simplify_mesh(V, F, FR, FF, VV,  np.float64(0.000050))
    n_FF, _ = igl.triangle_triangle_adjacency(n_F)
    n_k, n_FR, _, R_map = rebuild_region(n_FR, n_FF)

    try:
        n_VF, n_VV, _ = vertex_face_adjacency(V, n_F, n_FF)
        n_RV, n_RA, n_RR, n_topology_errors = compute_region_topology(n_F, n_k, n_FR, n_FF, n_VF, n_VV)
        print(f"After simplification: {n_k} proxies, topology errors: {n_topology_errors}")

        if n_topology_errors > 0:
            
            _, n_A, _ = compute_face_properties(V, n_F)
            n_FR = delete_invalid_regions(n_FR, n_RR, n_A)
            n_RV, n_RA, n_RR, n_topology_errors = compute_region_topology(n_F, n_k, n_FR, n_FF, n_VF, n_VV)
            print(f"After repair: topology errors: {n_topology_errors}")

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
    ps.register_surface_mesh("my_mesh_simplified", V, n_F)

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
def flood(mode, V, F, FF, A, W, S, FN, PN, PX):
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





# splitting and merging (proxy fitting is used) ---------------------------------------------------------------------------------------

'''
Fix topology
1. Merge: If the loops that have 1 neighbor
2. Split: If the loops that have 2 neighbors
'''
def update_proxies_by_topology(mode, VF, k, RV, RA, RR, FR, FE, PN, PX, V, F, g, A, W, FN):

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
    k, FR, PX, PN = merge_regions(mode, chunks, FN, FR, k, PX, PN, V, F, g, A, W)

    # split all faces
    k, FR, PX, PN = split_regions(mode, k, FR, PX, PN, V, F, g, A, W, FN, fids)

    return k, FR, PX, PN

'''
Fix local optimum: only when energy distribution is uneven / variance is large)
1. Merge: If two neighboring proxies have smallest error
2. Split: If a proxy has largest error
'''
def update_proxies_by_teleportation(mode, k, FR, FE, D, RR, PX, PN, V, F, g, A, W, FN):

    # Pair to merge
    min_normal_diff = np.inf
    min_pair = (-1, -1)
    pair_visited = set()
    
    # Region to split
    max_d = -1.0
    max_d_rid = -1

    for rid in range(k):
        # Find the proxy with the largest distortion
        if D[rid] > max_d:
            max_d = D[rid]
            max_d_rid = rid

        # Find the neighboring proxies with the smallest normal difference
        for loop in RR[rid]:
            for rid_neighbor in loop:

                if rid_neighbor == -1:
                    continue

                if (rid_neighbor, rid) in pair_visited:
                    continue

                pair_visited.add((rid, rid_neighbor))
                diff = np.sum((PN[rid] - PN[rid_neighbor]) ** 2)

                if diff < min_normal_diff:
                    min_normal_diff = diff
                    min_pair = (rid, rid_neighbor)

    '''
    # Split at anchor: this region may not have any anchors, so sometimes it can fail to split.
    vid_anchors = [
        RV[max_d_rid][loop_id][anchor_id]
        for loop_id, anchor_list in enumerate(RA[max_d_rid])
        for anchor_id in anchor_list
        ]
    
    fid_list = [fid for vid in vid_anchors for fid in VF[vid] if FR[fid] == max_d_rid]
    errors = np.array([FE[fid] for fid in fid_list])
    fid_max_error = fid_list[np.argmax(errors)]
    '''
    # Split at largest error face
    mask = FR == max_d_rid
    fid_list = np.where(mask)[0]
    fid_max_error = fid_list[np.argmax([FE[fid] for fid in fid_list])]

    k, FR, PX, PN = merge_regions(mode, [set(min_pair)], FN, FR, k, PX, PN, V, F, g, A, W)
    k, FR, PX, PN = split_regions(mode, k, FR, PX, PN, V, F, g, A, W, FN, [fid_max_error])

    return k, FR, PX, PN

# splitting
def split_regions(mode, k, FR, PX, PN, V, F, g, A, W, FN, fids):

    for fid in fids:
        old_rid = FR[fid]
       
        # Add a new proxy
        FR[fid] = k
        k += 1
        PX.append(g[fid])
        PN.append(FN[fid])

        # Update the proxy for the old region
        mask = FR == old_rid
        if mode == "L2":
            pn, px = vector_compute_proxy_L2(V, F, g, A, W, mask)
        elif mode == "L21":
            pn, px = vector_compute_proxy_L21(g, A, W, FN, mask)

        PN[old_rid] = pn
        PX[old_rid] = px

    return k, FR, PX, PN

# merging (chunks: list of sets)
def merge_regions(mode, rid_chunks, FN, FR, k, PX, PN, V, F, g, A, W):

    r_count = k
    id_deleted = []
    for chunk in rid_chunks:
        
        if not chunk:
            continue

        # compute new proxy for the merged region
        rid_list = list(chunk)
        mask = np.isin(FR, rid_list)
        if mode == "L2":
            new_pn, new_px = vector_compute_proxy_L2(V, F, g, A, W, mask)
        elif mode == "L21":
            new_pn, new_px = vector_compute_proxy_L21(g, A, W, FN, mask)

        # mark the old regions as deleted
        id_deleted.extend(rid_list)
        for rid in rid_list:
            PN[rid] = None
            PX[rid] = None

        # update FR, PX, PN
        FR[mask] = r_count
        PX.append(new_px)
        PN.append(new_pn)   
        r_count += 1

    # clean up regions
    new_PX = []
    new_PN = []
    new_FR = np.full(len(FR), -1, dtype=np.int32)

    new_rid = 0
    for i in range(r_count):

        if i in id_deleted:
            continue
        else:
            new_PX.append(PX[i])
            new_PN.append(PN[i])
            new_FR[FR == i] = new_rid
            new_rid += 1

    new_k = new_rid

    return new_k, new_FR, new_PX, new_PN





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
def vector_compute_proxy_L2(V, F, g, A, W, region_mask):

    g_i = g[region_mask]
    A_i = A[region_mask]
    W_i = W[region_mask]
    AW_i = A_i * W_i
    RA_j = AW_i.sum()
    PX_j = (AW_i[:, np.newaxis] * g_i).sum(axis=0) / RA_j

    if RA_j == 0:
        return None, None

    # Need to be double checked
    Q_paper = np.array([[10,7,0],
                        [7,10,0],
                        [0,0,0]])
    
    Q_exact = np.array([[2,-1,0],
                        [-1,2,0],
                        [0,0,0]])

    # Compute covariance matrix
    M_cov = np.zeros((3,3))

    V1 = V[F[region_mask, 0]]
    V2 = V[F[region_mask, 1]] 
    V3 = V[F[region_mask, 2]]
    
    e1 = V2 - V1
    e2 = V3 - V1

    M = np.zeros((len(e1), 3, 3))
    M[:, 0, :] = e1
    M[:, 1, :] = e2

    MTQM = M.transpose(0,2,1) @ Q_exact @ M
    ggT = g_i[:, :, None] @ g_i[:, None, :]
    M_cov = (AW_i[:, np.newaxis, np.newaxis] *((1 / 36) * MTQM + ggT)).sum(axis=0) - RA_j * (PX_j[:, None] @ PX_j[None, :])

    # Compute the eigenvector corresponding to the smallest eigenvalue
    eigvals, eigvecs = np.linalg.eigh(M_cov)
    id_min = np.argmin(eigvals)
    PN_j = eigvecs[:, id_min] # [:,_] means all rows
    PN_j /= np.linalg.norm(PN_j) # normalize the vector

    return PN_j, PX_j

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

# Sobolev H1 metric
_lambda = 1.0 # (0 < lambda < infinity)
def compute_error_H1(V, F, A, W, FN, fid, PN_j, PX_j): # W_i is the weight for fid
   E_i = compute_error_L2(V, F, A, W, fid, PN_j, PX_j) + _lambda * compute_error_L21(A, W, FN, fid, PN_j)
   return E_i
# root solver needed
def vector_compute_proxy_H1(V, F, g, A, W, FN, region_mask):
    """
    g_i = g[region_mask]
    A_i = A[region_mask]
    W_i = W[region_mask]
    AW_i = A_i * W_i
    RA_j = AW_i.sum()
    PX_j = (AW_i[:, np.newaxis] * g_i).sum(axis=0) / RA_j

    if RA_j == 0:
        return None, None

    # Need to be double checked
    Q_paper = np.array([[10,7,0],
                        [7,10,0],
                        [0,0,0]])
    
    Q_exact = np.array([[2,-1,0],
                        [-1,2,0],
                        [0,0,0]])

    # Compute covariance matrix
    M_cov = np.zeros((3,3))

    V1 = V[F[region_mask, 0]]
    V2 = V[F[region_mask, 1]] 
    V3 = V[F[region_mask, 2]]
    
    e1 = V2 - V1
    e2 = V3 - V1

    M = np.zeros((len(e1), 3, 3))
    M[:, 0, :] = e1
    M[:, 1, :] = e2

    MTQM = M.transpose(0,2,1) @ Q_exact @ M
    ggT = g_i[:, :, None] @ g_i[:, None, :]
    M_cov = (AW_i[:, np.newaxis, np.newaxis] *((1 / 36) * MTQM + ggT)).sum(axis=0) - RA_j * (PX_j[:, None] @ PX_j[None, :])

    # Compute the eigenvector corresponding to the smallest eigenvalue
    eigvals, eigvecs = np.linalg.eigh(M_cov)
    id_min = np.argmin(eigvals)
    PN_j = eigvecs[:, id_min] # [:,_] means all rows
    PN_j /= np.linalg.norm(PN_j) # normalize the vector
    """

    pass




# AOGP (ellipsoid proxy) ==============================================================================================================





# Construct discrete delauney triangulation (CDT) for each region ---------------------------------------------------------------------

def simplify_mesh(V, F, FR, FF, VV, collapse_threshold=np.float64(0.0)):
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
    # Interestingly, when the mesh is too small or big, float32 may lead to precision issues that stops the queue poping out the correct result in eager dijkstra, but the lazy dijkstra is more robust. 
    seeds = np.full(len(V), -1, dtype=np.int32)
    dists = np.full(len(V), np.inf, dtype=np.float64)
    for vid in range(len(V)):
        if len(vv_boundary_adj[vid]) > 2:
            seeds[vid] = vid
            dists[vid] = 0.0

# 2. flood boundary vertices from anchors (dijkstra)
    seeds_constraint, dists_constraint = dijkstra_voronoi(V, vv_boundary_adj, seeds, dists)

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
    chunk_heads = []
    for chunk in seed_merge_chunks:
        if chunk:
            head = min(chunk)
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
    seeds_final, _ = dijkstra_voronoi(V, VV, seeds_constraint, dists_constraint)

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

    '''
    # Debug visualization
    unassigned_vids = np.where(seeds_final == -1)[0]
    print(f"Unassigned vertices: {len(unassigned_vids)}")

    # Register unassigned vertices
    if len(unassigned_vids) > 0:
        ps.register_point_cloud(
            "unassigned_vertices",
            V[unassigned_vids]
        )
    '''
    return np.array(new_F, dtype=np.int32), np.array(new_FR, dtype=np.int32)
                
def dijkstra_voronoi(V, vv_adj, seeds_constraint, min_dist_constraint):

    min_seeds = seeds_constraint
    min_dists = min_dist_constraint

    constraint_mask = min_seeds != -1
    
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
                min_seeds[vid_neighbor] = seed_vid  # Assign the seed ID to the vertex
                min_dists[vid_neighbor] = new_dist  # Update the minimum distance for the neighbor
                heapq.heappush(pq, (new_dist, vid_neighbor, seed_vid))  # Push the neighbor onto the priority queue

    return min_seeds, min_dists    








# Main function
def main():
    seed_array = np.random.randint(0, 65000, size=140)
    VSA("L21", "C:\\Users\\yangw\\Desktop\\CG_projects\\vsa_project\\stanford-bunny.obj", seed_array)

if __name__ == "__main__":
    main()
