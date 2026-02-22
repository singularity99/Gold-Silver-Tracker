import json
import os
from datetime import datetime

def _data_file() -> str:
    """Resolve writable path for portfolio data. Falls back to /tmp on read-only filesystems."""
    primary = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio_data.json")
    try:
        with open(primary, "a"):
            pass
        return primary
    except OSError:
        return os.path.join("/tmp", "portfolio_data.json")


def _load() -> dict:
    path = _data_file()
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"total_pot": 2_000_000, "purchases": []}


def _save(data: dict):
    with open(_data_file(), "w") as f:
        json.dump(data, f, indent=2, default=str)


def get_portfolio() -> dict:
    return _load()


def set_total_pot(amount: float):
    data = _load()
    data["total_pot"] = amount
    _save(data)


def add_purchase(metal: str, etc_ticker: str, gbp_amount: float,
                 price_per_unit: float, quantity: float, notes: str = ""):
    data = _load()
    purchase = {
        "id": len(data["purchases"]) + 1,
        "date": datetime.now().isoformat(),
        "metal": metal,
        "etc_ticker": etc_ticker,
        "gbp_amount": gbp_amount,
        "price_per_unit": price_per_unit,
        "quantity": quantity,
        "notes": notes,
    }
    data["purchases"].append(purchase)
    _save(data)
    return purchase


def delete_purchase(purchase_id: int):
    data = _load()
    data["purchases"] = [p for p in data["purchases"] if p["id"] != purchase_id]
    _save(data)


def get_summary(current_etc_prices: dict) -> dict:
    """Compute portfolio summary with live P&L."""
    data = _load()
    total_pot = data["total_pot"]
    purchases = data["purchases"]
    total_deployed = sum(p["gbp_amount"] for p in purchases)
    remaining = total_pot - total_deployed

    positions = {}
    for p in purchases:
        ticker = p["etc_ticker"]
        if ticker not in positions:
            positions[ticker] = {"qty": 0, "cost": 0, "metal": p["metal"]}
        positions[ticker]["qty"] += p["quantity"]
        positions[ticker]["cost"] += p["gbp_amount"]

    total_current_value = 0
    position_details = []
    for ticker, pos in positions.items():
        current_price = current_etc_prices.get(ticker, {}).get("price", 0)
        currency = current_etc_prices.get(ticker, {}).get("currency", "GBp")
        if currency == "GBp":
            current_price_gbp = current_price / 100.0
        else:
            current_price_gbp = current_price
        current_value = pos["qty"] * current_price_gbp
        pnl = current_value - pos["cost"]
        pnl_pct = (pnl / pos["cost"] * 100) if pos["cost"] > 0 else 0
        total_current_value += current_value
        position_details.append({
            "ticker": ticker,
            "metal": pos["metal"],
            "quantity": pos["qty"],
            "cost_gbp": pos["cost"],
            "current_value_gbp": current_value,
            "pnl_gbp": pnl,
            "pnl_pct": pnl_pct,
        })

    total_pnl = total_current_value - total_deployed

    return {
        "total_pot": total_pot,
        "total_deployed": total_deployed,
        "remaining": remaining,
        "total_current_value": total_current_value,
        "total_pnl": total_pnl,
        "total_pnl_pct": (total_pnl / total_deployed * 100) if total_deployed > 0 else 0,
        "positions": position_details,
        "purchases": purchases,
        "deployment_pct": (total_deployed / total_pot * 100) if total_pot > 0 else 0,
    }
