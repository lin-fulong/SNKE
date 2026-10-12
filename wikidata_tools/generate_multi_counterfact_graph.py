import argparse
import json
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.error import URLError


WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
USER_AGENT = "SNKE multi_counterfact_graph generator/1.0"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_relation_label_map(
    counterfact: List[dict],
    counterfact_graph: List[dict],
    mquake: List[dict],
    mquake_graph: List[dict],
) -> Dict[str, str]:
    relation_id_to_label: Dict[str, str] = {}

    for record, graph_record in zip(counterfact, counterfact_graph):
        rewrite = record["requested_rewrite"]
        relation_id_to_label.setdefault(
            rewrite["relation_id"], graph_record["triples"][0]["relation"]
        )

    for record, graph_record in zip(mquake, mquake_graph):
        for rewrite, triples in zip(record["requested_rewrite"], graph_record["triples"]):
            relation_id_to_label.setdefault(rewrite["relation_id"], triples[0]["relation"])

    return relation_id_to_label


def build_target_neighbor_cache(
    counterfact: List[dict],
    counterfact_graph: List[dict],
    mquake: List[dict],
    mquake_graph: List[dict],
) -> Dict[str, List[dict]]:
    target_to_neighbors: Dict[str, List[dict]] = {}

    for record, graph_record in zip(counterfact, counterfact_graph):
        target_id = record["requested_rewrite"]["target_new"]["id"]
        target_to_neighbors.setdefault(target_id, graph_record["triples"][1:])

    for record, graph_record in zip(mquake, mquake_graph):
        for rewrite, triples in zip(record["requested_rewrite"], graph_record["triples"]):
            target_id = rewrite["target_new"]["id"]
            target_to_neighbors.setdefault(target_id, triples[1:])

    return target_to_neighbors


def fetch_neighbor_triples(
    entity_id: str,
    entity_label: str,
    timeout: int = 60,
    max_retries: int = 6,
) -> List[dict]:
    query = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?pLabel ?oLabel WHERE {{
  BIND(wd:{entity_id} AS ?s)
  ?s ?p ?o .
  ?wdProp wikibase:directClaim ?p .

  FILTER(STRSTARTS(STR(?p), STR(wdt:)))
  FILTER(STRSTARTS(STR(?o), STR(wd:)))
  FILTER(!STRSTARTS(STR(?p), STR(wdt:P910)))
  FILTER(!STRSTARTS(STR(?p), STR(wdt:P1423)))

  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "en" .
    ?wdProp rdfs:label ?pLabel .
    ?o rdfs:label ?oLabel .
  }}
}}
"""
    params = urllib.parse.urlencode({"query": query, "format": "json"})
    request = urllib.request.Request(
        f"{WIKIDATA_SPARQL_URL}?{params}",
        headers={
            "Accept": "application/sparql-results+json",
            "User-Agent": USER_AGENT,
        },
    )

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except (URLError, TimeoutError, ssl.SSLError) as exc:
            last_error = exc
            if attempt == max_retries:
                raise
            sleep_for = min(2 ** (attempt - 1), 20)
            print(
                f"  Request failed for {entity_id} on attempt {attempt}/{max_retries}: {exc}. "
                f"Retrying in {sleep_for}s..."
            )
            time.sleep(sleep_for)
    else:
        raise RuntimeError(f"Failed to fetch {entity_id}: {last_error}")

    triples: List[dict] = []
    for row in payload["results"]["bindings"]:
        triples.append(
            {
                "subject": entity_label,
                "relation": row["pLabel"]["value"],
                "target": row["oLabel"]["value"],
            }
        )
    return triples


def normalize_target_cache(raw_cache: dict) -> Dict[str, List[dict]]:
    normalized: Dict[str, List[dict]] = {}
    for key, value in raw_cache.items():
        if isinstance(value, list):
            normalized[key] = value
    return normalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing counterfact/mquake/multi_counterfact JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/multi_counterfact_graph.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--target-cache",
        type=Path,
        default=Path("data/multi_counterfact_target_neighbors_cache.json"),
        help="Cache path for per-target neighbor triples.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.25,
        help="Delay between live Wikidata requests.",
    )
    args = parser.parse_args()

    counterfact = load_json(args.data_dir / "counterfact.json")
    counterfact_graph = load_json(args.data_dir / "counterfact_graph.json")
    mquake = load_json(args.data_dir / "mquake.json")
    mquake_graph = load_json(args.data_dir / "mquake_graph.json")
    multi_counterfact = load_json(args.data_dir / "multi_counterfact.json")

    relation_id_to_label = build_relation_label_map(
        counterfact, counterfact_graph, mquake, mquake_graph
    )
    target_to_neighbors = build_target_neighbor_cache(
        counterfact, counterfact_graph, mquake, mquake_graph
    )

    if args.target_cache.exists():
        target_to_neighbors.update(normalize_target_cache(load_json(args.target_cache)))

    missing_relation_ids = sorted(
        {
            record["requested_rewrite"]["relation_id"]
            for record in multi_counterfact
            if record["requested_rewrite"]["relation_id"] not in relation_id_to_label
        }
    )
    if missing_relation_ids:
        raise ValueError(f"Missing relation labels for: {missing_relation_ids}")

    required_targets: Dict[str, str] = {}
    for record in multi_counterfact:
        rewrite = record["requested_rewrite"]
        required_targets[rewrite["target_new"]["id"]] = rewrite["target_new"]["str"]

    missing_target_ids = sorted(
        [target_id for target_id in required_targets if target_id not in target_to_neighbors]
    )
    print(f"Need live Wikidata fetch for {len(missing_target_ids)} target entities")

    for index, target_id in enumerate(missing_target_ids, start=1):
        target_label = required_targets[target_id]
        print(f"[{index}/{len(missing_target_ids)}] Fetching neighbors for {target_id} ({target_label})")
        triples = fetch_neighbor_triples(target_id, target_label)
        target_to_neighbors[target_id] = triples
        dump_json(args.target_cache, target_to_neighbors)
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    output_records: List[dict] = []
    for record in multi_counterfact:
        rewrite = record["requested_rewrite"]
        relation_label = relation_id_to_label[rewrite["relation_id"]]
        target_id = rewrite["target_new"]["id"]
        triples = [
            {
                "subject": rewrite["subject"],
                "relation": relation_label,
                "target": rewrite["target_new"]["str"],
            }
        ]
        triples.extend(target_to_neighbors.get(target_id, []))
        output_records.append({"case_id": record["case_id"], "triples": triples})

    dump_json(args.output, output_records)
    print(f"Wrote {len(output_records)} records to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
