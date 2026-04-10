import sys
import pathlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ROOT = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))


def run_analysis():
    try:
        import archiver
        import validator
    except ImportError:
        from backend import archiver, validator

    client = archiver._get_collection()
    validator.warm_up()

    print("\n" + "=" * 50)
    print(" 2026 SEMANTIC AUTOMATION SUITE: RUNNING ")
    print("=" * 50)

    # Fetch Peace Centroid (Run 32 -- Board of Peace)
    peace_res = client.get(where={"word": {"$in": ["שלומ", "שלוה", "הסכמ"]}}, include=["embeddings"])
    dark_res = client.get(where={"word": {"$in": ["החיה", "תרסו", "אנטיכריסט", "ארמילוס"]}}, include=["embeddings"])

    if len(peace_res["embeddings"]) == 0 or len(dark_res["embeddings"]) == 0:
        print(f"!! Error: Peace ({len(peace_res['embeddings'])}) or Dark ({len(dark_res['embeddings'])}) clusters missing.")
        return

    peace_centroid = np.mean(peace_res["embeddings"], axis=0).reshape(1, -1)
    dark_centroid = np.mean(dark_res["embeddings"], axis=0).reshape(1, -1)
    duality_dist = 1 - cosine_similarity(peace_centroid, dark_centroid)[0][0]

    # Fetch Zion Centroid from archive (Runs 8/21)
    zion_res = client.get(where={"word": {"$in": ["ציונ", "ציון"]}}, include=["embeddings"])

    # Haifa cluster -- embed directly via HeBERT (Run 35 archived 0 entries)
    haifa_concepts = ["חיפה", "תבערה", "תשפו"]
    haifa_vecs = np.vstack([validator._embed(w).numpy() for w in haifa_concepts])
    haifa_centroid = np.mean(haifa_vecs, axis=0).reshape(1, -1)

    print(f"\n[ARCHIVE STATS]")
    print(f"Total Entries: {client.count()}")
    print(f"Peace Samples: {len(peace_res['embeddings'])} | Dark Samples: {len(dark_res['embeddings'])}")

    print(f"\n[DUALITY ANALYSIS]")
    print(f"Peace vs Dark (Duality Gap): {duality_dist:.4f}")

    print(f"\n[HAIFA-ZION DISTANCE TEST]")
    if len(zion_res["embeddings"]) > 0:
        zion_centroid = np.mean(zion_res["embeddings"], axis=0).reshape(1, -1)
        haifa_dist = 1 - cosine_similarity(zion_centroid, haifa_centroid)[0][0]
        print(f"Zion Samples: {len(zion_res['embeddings'])}")
        print(f"Disaster Gap (Haifa vs Zion): {haifa_dist:.4f}")
    else:
        haifa_dist = 1 - cosine_similarity(peace_centroid, haifa_centroid)[0][0]
        print(f"[Note: No Zion entries in archive -- using Peace centroid as proxy]")
        print(f"Disaster Gap (Haifa vs Peace/Zion proxy): {haifa_dist:.4f}")

    if haifa_dist < 0.35:
        print("VERDICT: INTERNAL COLLISION. The code views the Haifa fire as a component of the 2026 Peace state.")
    elif haifa_dist > 0.55:
        print("VERDICT: EXTERNAL DISRUPTION. The code views the fire as an outside event attacking the Peace state.")
    else:
        print("VERDICT: SEMANTIC NEUTRAL. Standard thematic separation.")

    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_analysis()