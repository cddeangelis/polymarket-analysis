import pandas as pd
from polymarket_analysis.data import _parse_event


def _make_event(
    id="evt1",
    timestamp="1700000000",
    maker="0xmaker",
    taker="0xtaker",
    maker_asset="0",
    taker_asset="token_yes",
    maker_amount="1000000",
    taker_amount="500000",
):
    return {
        "id": id,
        "timestamp": timestamp,
        "maker": maker,
        "taker": taker,
        "makerAssetId": maker_asset,
        "takerAssetId": taker_asset,
        "makerAmountFilled": maker_amount,
        "takerAmountFilled": taker_amount,
    }


class TestParseEvent:
    def test_buy_side(self):
        event = _make_event(
            maker_asset="0",
            taker_asset="token_yes",
            maker_amount="5000000",
        )
        result = _parse_event(event)

        assert result["side"] == "buy"
        assert result["token_id"] == "token_yes"
        assert result["amount_usd"] == 5.0
        assert result["wallet"] == "0xtaker"
        assert result["id"] == "evt1"

    def test_sell_side(self):
        event = _make_event(
            maker_asset="token_no",
            taker_asset="0",
            taker_amount="3000000",
        )
        result = _parse_event(event)

        assert result["side"] == "sell"
        assert result["token_id"] == "token_no"
        assert result["amount_usd"] == 3.0

    def test_timestamp_parsed_as_int(self):
        event = _make_event(timestamp="1700000999")
        result = _parse_event(event)
        assert result["timestamp"] == 1700000999

    def test_fractional_usd(self):
        event = _make_event(maker_asset="0", maker_amount="1500000")
        result = _parse_event(event)
        assert result["amount_usd"] == 1.5


class TestDeduplication:
    def test_dedup_removes_duplicate_ids(self):
        events = [
            _make_event(id="dup1", maker_asset="0", taker_asset="token_yes"),
            _make_event(id="dup1", maker_asset="token_yes", taker_asset="0"),
            _make_event(id="unique", maker_asset="0", taker_asset="token_no"),
        ]
        records = [_parse_event(e) for e in events]
        df = pd.DataFrame(records)
        df = df.drop_duplicates(subset=["id"]).drop(columns=["id"])
        assert len(df) == 2
