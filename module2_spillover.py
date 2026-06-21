"""
ParkIQ — Module 2: Spillover Chain Mapper
Builds a directed graph: high-stress junction → nearby lower-stress junctions.
Uses GeoPandas + NetworkX. Computes spillover (PageRank) centrality.
"""
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
from config import SCORED_PARQUET, JUNCTION_STRESS, SPILLOVER_JSON, SPILLOVER_RADIUS_M


def run():
    print("[Module 2] Building spillover chain graph …")
    df        = pd.read_parquet(SCORED_PARQUET)
    stress_df = pd.read_parquet(JUNCTION_STRESS)

    # Junction centroids
    coords = (df[df["junction_name"] != "No Junction"]
              .groupby("junction_name")[["latitude","longitude"]]
              .mean().reset_index())
    coords.columns = ["junction_name","lat","lon"]

    merged = coords.merge(
        stress_df[["junction_name","predicted_stress_next_h","total_violations"]],
        on="junction_name", how="left"
    ).fillna({"predicted_stress_next_h": 0, "total_violations": 0})

    gdf = gpd.GeoDataFrame(
        merged,
        geometry=[Point(r.lon, r.lat) for _, r in merged.iterrows()],
        crs="EPSG:4326"
    ).to_crs("EPSG:32643")   # metres (UTM 43N — Karnataka)

    G = nx.DiGraph()
    for _, row in merged.iterrows():
        G.add_node(row["junction_name"],
                   lat=float(row["lat"]), lon=float(row["lon"]),
                   stress=float(row["predicted_stress_next_h"]),
                   total_violations=int(row["total_violations"]))

    # Edges within SPILLOVER_RADIUS_M, directed high→low stress
    for i, ri in gdf.iterrows():
        for j, rj in gdf.iterrows():
            if i == j: continue
            dist = ri.geometry.distance(rj.geometry)
            if dist <= SPILLOVER_RADIUS_M:
                if ri["predicted_stress_next_h"] >= rj["predicted_stress_next_h"]:
                    G.add_edge(ri["junction_name"], rj["junction_name"],
                               distance_m=round(float(dist), 1),
                               weight=round(1 / (dist + 1), 6))

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # PageRank spillover centrality
    pr = nx.pagerank(G, weight="weight") if G.number_of_nodes() > 0 else {}
    nx.set_node_attributes(G, pr, "spillover_centrality")

    data = nx.node_link_data(G)
    with open(SPILLOVER_JSON, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved → {SPILLOVER_JSON}")

    # Print top spillover nodes
    top = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:5]
    for name, score in top:
        print(f"  {name[:55]:55s}  centrality={score:.4f}")

    return G, pr


if __name__ == "__main__":
    run()
