import json
import os
import base64
from datetime import datetime
from urllib.request import Request, urlopen
from urllib.error import URLError

REPO_FILE_PATH = "portfolio_data.json"
DEFAULT_DATA = {"total_pot": 2_000_000, "purchases": [], "tf_weights": {"Short": 48, "Medium": 37, "Long": 15}}


def _github_config() -> dict | None:
    """Load GitHub API config from Streamlit secrets or environment."""
    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPO", "")
    if not token or not repo:
        try:
            import streamlit as st
            token = st.secrets.get("GITHUB_TOKEN", "")
            repo = st.secrets.get("GITHUB_REPO", "")
        except Exception:
            pass
    if token and repo:
        return {"token": token, "repo": repo}
    return None


def _github_read(config: dict) -> tuple[dict, str | None]:
    """Read portfolio data from GitHub repo. Returns (data, sha)."""
    url = f"https://api.github.com/repos/{config['repo']}/contents/{REPO_FILE_PATH}"
    req = Request(url, headers={
        "Authorization": f"token {config['token']}",
        "Accept": "application/vnd.github.v3+json",
    })
    try:
        with urlopen(req) as resp:
            result = json.loads(resp.read())
            content = base64.b64decode(result["content"]).decode("utf-8")
            return json.loads(content), result["sha"]
    except URLError:
        return DEFAULT_DATA.copy(), None


def _github_write(config: dict, data: dict, sha: str | None):
    """Write portfolio data to GitHub repo via API."""
    url = f"https://api.github.com/repos/{config['repo']}/contents/{REPO_FILE_PATH}"
    content_b64 = base64.b64encode(json.dumps(data, indent=2, default=str).encode()).decode()
    body = {
        "message": f"Portfolio update {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "content": content_b64,
    }
    if sha:
        body["sha"] = sha
    payload = json.dumps(body).encode()
    req = Request(url, data=payload, method="PUT", headers={
        "Authorization": f"token {config['token']}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    })
    try:
        with urlopen(req) as resp:
            resp.read()
    except URLError:
        pass


def _local_file() -> str:
    """Resolve writable local path for portfolio data."""
    primary = os.path.join(os.path.dirname(os.path.abspath(__file__)), REPO_FILE_PATH)
    try:
        with open(primary, "a"):
            pass
        return primary
    except OSError:
        return os.path.join("/tmp", REPO_FILE_PATH)


def _load() -> dict:
    gh = _github_config()
    if gh:
        data, _ = _github_read(gh)
        return data
    path = _local_file()
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            pass
    return DEFAULT_DATA.copy()


def _save(data: dict):
    gh = _github_config()
    if gh:
        _, sha = _github_read(gh)
        _github_write(gh, data, sha)
    path = _local_file()
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def get_portfolio() -> dict:
    return _load()


def set_total_pot(amount: float):
    data = _load()
    data["total_pot"] = amount
    _save(data)


def get_tf_weights() -> dict:
    data = _load()
    return data.get("tf_weights", DEFAULT_DATA["tf_weights"])


def set_tf_weights(weights: dict):
    data = _load()
    data["tf_weights"] = weights
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


def export_portfolio_json() -> str:
    """Export portfolio data as JSON string (for download)."""
    return json.dumps(_load(), indent=2, default=str)


def import_portfolio_json(json_str: str):
    """Import portfolio data from JSON string (from upload)."""
    data = json.loads(json_str)
    if "total_pot" in data and "purchases" in data:
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
