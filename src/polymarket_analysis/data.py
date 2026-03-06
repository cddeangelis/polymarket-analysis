import json
import time
import requests
import pandas as pd

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
GOLDSKY_SUBGRAPH = (
    "https://api.goldsky.com/api/public/"
    "project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn"
)

ORDER_FILLED_BY_MAKER_QUERY = """
query($token_ids: [String!]!, $last_id: String!, $page_size: Int!) {
  orderFilledEvents(
    first: $page_size,
    where: {
      makerAssetId_in: $token_ids,
      id_gt: $last_id
    }
  ) {
    id
    timestamp
    maker
    taker
    makerAssetId
    takerAssetId
    makerAmountFilled
    takerAmountFilled
  }
}
"""

ORDER_FILLED_BY_TAKER_QUERY = """
query($token_ids: [String!]!, $last_id: String!, $page_size: Int!) {
  orderFilledEvents(
    first: $page_size,
    where: {
      takerAssetId_in: $token_ids,
      id_gt: $last_id
    }
  ) {
    id
    timestamp
    maker
    taker
    makerAssetId
    takerAssetId
    makerAmountFilled
    takerAmountFilled
  }
}
"""


def get_market_info(slug: str) -> dict:
    resp = requests.get(f"{GAMMA_API_BASE}/markets/slug/{slug}")
    resp.raise_for_status()
    market = resp.json()

    # API returns clobTokenIds and outcomes as JSON strings
    token_ids = json.loads(market.get("clobTokenIds", "[]"))
    outcomes = json.loads(market.get("outcomes", "[]"))

    token_map = dict(zip(outcomes, token_ids))
    yes_token_id = token_map.get("Yes")
    no_token_id = token_map.get("No")

    if not yes_token_id or not no_token_id:
        raise ValueError(f"Could not find Yes/No tokens for market '{slug}'")

    return {
        "question": market["question"],
        "yes_token_id": yes_token_id,
        "no_token_id": no_token_id,
        "condition_id": market.get("conditionId", ""),
    }


def _run_subgraph_query(query: str, token_ids: list[str]) -> list[dict]:
    all_events = []
    last_id = ""
    page_size = 1000

    while True:
        for attempt in range(4):
            try:
                resp = requests.post(
                    GOLDSKY_SUBGRAPH,
                    json={
                        "query": query,
                        "variables": {
                            "token_ids": token_ids,
                            "last_id": last_id,
                            "page_size": page_size,
                        },
                    },
                    timeout=30,
                )
                resp.raise_for_status()
            except requests.exceptions.RequestException as exc:
                page_size = max(100, page_size // 2)
                wait = 2 ** attempt
                print(f"  Request error ({exc}), reducing page size to {page_size}, retrying in {wait}s (attempt {attempt + 1}/4)...")
                time.sleep(wait)
                continue

            data = resp.json()

            if "errors" in data:
                err_msg = data["errors"][0].get("message", "")
                if "timed out" in err_msg.lower() or "timeout" in err_msg.lower():
                    page_size = max(100, page_size // 2)
                    wait = 2 ** attempt
                    print(f"  Subgraph timeout, reducing page size to {page_size}, retrying in {wait}s (attempt {attempt + 1}/4)...")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Subgraph query error: {data['errors']}")

            break  # success
        else:
            raise RuntimeError("Subgraph query failed after 4 attempts")

        events = data["data"]["orderFilledEvents"]
        if not events:
            break

        all_events.extend(events)
        last_id = events[-1]["id"]
        print(f"  Fetched {len(all_events)} events so far...")

        # Gradually restore page size after a successful reduced-size fetch
        if page_size < 1000:
            page_size = min(1000, page_size * 2)

    return all_events


def _parse_event(event: dict) -> dict:
    maker_asset = event["makerAssetId"]

    # makerAssetId == "0" means maker is giving USDC (buying tokens) -> BUY
    # makerAssetId != "0" means maker is giving tokens (selling tokens) -> SELL
    if maker_asset == "0":
        side = "buy"
        amount_raw = int(event["makerAmountFilled"])
        token_id = event["takerAssetId"]
    else:
        side = "sell"
        amount_raw = int(event["takerAmountFilled"])
        token_id = maker_asset

    return {
        "id": event["id"],
        "timestamp": int(event["timestamp"]),
        "wallet": event["maker"],
        "side": side,
        "token_id": token_id,
        "amount_usd": amount_raw / 1e6,
    }


def fetch_order_fills(yes_token_id: str, no_token_id: str) -> pd.DataFrame:
    token_ids = [yes_token_id, no_token_id]

    print("Fetching orders where makerAssetId matches tokens (SELL side)...")
    maker_events = _run_subgraph_query(ORDER_FILLED_BY_MAKER_QUERY, token_ids)

    print("Fetching orders where takerAssetId matches tokens (BUY side)...")
    taker_events = _run_subgraph_query(ORDER_FILLED_BY_TAKER_QUERY, token_ids)

    all_events = maker_events + taker_events

    if not all_events:
        raise ValueError("No order fill events found for these tokens")

    records = [_parse_event(e) for e in all_events]
    df = pd.DataFrame(records)

    # Deduplicate — a trade between two tracked tokens appears in both queries
    df = df.drop_duplicates(subset=["id"]).drop(columns=["id"])

    # Tag outcome based on token ID
    df["outcome"] = df["token_id"].map(
        {yes_token_id: "yes", no_token_id: "no"}
    )
    # Drop any events that don't match our tokens (shouldn't happen but be safe)
    df = df.dropna(subset=["outcome"])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.drop(columns=["token_id"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Total trades: {len(df)}")
    print(f"Unique wallets: {df['wallet'].nunique()}")
    print(f"Total volume: ${df['amount_usd'].sum():,.0f}")

    return df
