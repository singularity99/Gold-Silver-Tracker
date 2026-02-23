import json
import os
import base64
from copy import deepcopy
from datetime import datetime
from urllib.request import Request, urlopen
from urllib.error import URLError

REPO_FILE_PATH = "portfolio_data.json"
DEFAULT_TF_WEIGHTS = {"Short": 48, "Medium": 37, "Long": 15}
DEFAULT_CONFIG = {
    "selected_etcs": ["SGLN.L", "SSLN.L"],
    "fib_tolerance": 2.0,
    "gs_ratio_threshold": 63.0,
    "rsi_period": 14,
    "ema_fast": 9,
    "ema_slow": 21,
    "sma_fast": 20,
    "sma_slow": 50,
    "whale_vol_threshold": 2.0,
    "tf_weights": deepcopy(DEFAULT_TF_WEIGHTS),
    "profiles": {},
}
DEFAULT_DATA = {
    "total_pot": 2_000_000,
    "purchases": [],
    "tf_weights": deepcopy(DEFAULT_TF_WEIGHTS),
    "config": deepcopy(DEFAULT_CONFIG),
    "config_updated_at": "",
}


def _normalise_tf_weights(weights: dict | None) -> dict:
    base = deepcopy(DEFAULT_TF_WEIGHTS)
    if isinstance(weights, dict):
        for tf in ("Short", "Medium", "Long"):
            if tf in weights:
                try:
                    base[tf] = max(0, int(weights[tf]))
                except (TypeError, ValueError):
                    pass
    total = sum(base.values())
    if total == 100:
        return base
    if total <= 0:
        return deepcopy(DEFAULT_TF_WEIGHTS)
    short = int(round(base["Short"] * 100 / total))
    medium = int(round(base["Medium"] * 100 / total))
    long = 100 - short - medium
    return {"Short": short, "Medium": medium, "Long": long}


def _normalise_config(config: dict | None) -> dict:
    merged = deepcopy(DEFAULT_CONFIG)
    if isinstance(config, dict):
        if isinstance(config.get("selected_etcs"), list):
            merged["selected_etcs"] = [str(x) for x in config["selected_etcs"] if str(x).strip()]

        for key in ("fib_tolerance", "gs_ratio_threshold", "whale_vol_threshold"):
            if key in config:
                try:
                    merged[key] = float(config[key])
                except (TypeError, ValueError):
                    pass

        for key in ("rsi_period", "ema_fast", "ema_slow", "sma_fast", "sma_slow"):
            if key in config:
                try:
                    merged[key] = int(config[key])
                except (TypeError, ValueError):
                    pass

        merged["tf_weights"] = _normalise_tf_weights(config.get("tf_weights"))
        profiles = config.get("profiles")
        if isinstance(profiles, dict):
            cleaned_profiles = {}
            for name, profile_cfg in profiles.items():
                if not str(name).strip() or not isinstance(profile_cfg, dict):
                    continue
                cleaned_profiles[str(name).strip()] = _normalise_config({**profile_cfg, "profiles": {}})
            merged["profiles"] = cleaned_profiles
    else:
        merged["tf_weights"] = _normalise_tf_weights(None)
    return merged


def _merge_config(base: dict, updates: dict) -> dict:
    merged = deepcopy(base)
    if not isinstance(updates, dict):
        return merged
    if "selected_etcs" in updates and isinstance(updates["selected_etcs"], list):
        merged["selected_etcs"] = [str(x) for x in updates["selected_etcs"] if str(x).strip()]
    for key in ("fib_tolerance", "gs_ratio_threshold", "whale_vol_threshold"):
        if key in updates:
            try:
                merged[key] = float(updates[key])
            except (TypeError, ValueError):
                pass
    for key in ("rsi_period", "ema_fast", "ema_slow", "sma_fast", "sma_slow"):
        if key in updates:
            try:
                merged[key] = int(updates[key])
            except (TypeError, ValueError):
                pass
    if "tf_weights" in updates:
        merged["tf_weights"] = _normalise_tf_weights(updates.get("tf_weights"))
    if "profiles" in updates and isinstance(updates["profiles"], dict):
        cleaned_profiles = {}
        for name, profile_cfg in updates["profiles"].items():
            if not str(name).strip() or not isinstance(profile_cfg, dict):
                continue
            cleaned_profiles[str(name).strip()] = _normalise_config({**profile_cfg, "profiles": {}})
        merged["profiles"] = cleaned_profiles
    return _normalise_config(merged)


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


def _github_read(config: dict) -> tuple[dict | None, str | None]:
    """Read portfolio data from GitHub repo. Returns (data, sha) or (None, None) on failure."""
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
        return None, None


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


def _read_local_data() -> dict | None:
    path = _local_file()
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def _version_epoch(data: dict | None) -> float:
    if not isinstance(data, dict):
        return float("-inf")
    ver = data.get("config_updated_at", "")
    if not isinstance(ver, str) or not ver:
        return float("-inf")
    try:
        return datetime.fromisoformat(ver.replace("Z", "+00:00")).timestamp()
    except Exception:
        return float("-inf")


def _load() -> dict:
    local_data = _read_local_data()
    gh = _github_config()
    if gh:
        remote_data, _ = _github_read(gh)
        if remote_data and local_data:
            return local_data if _version_epoch(local_data) > _version_epoch(remote_data) else remote_data
        if remote_data:
            return remote_data
    if local_data:
        return local_data
    return deepcopy(DEFAULT_DATA)


def _save(data: dict):
    path = _local_file()
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    gh = _github_config()
    if gh:
        _, sha = _github_read(gh)
        _github_write(gh, data, sha)


def get_portfolio() -> dict:
    return _load()


def set_total_pot(amount: float):
    data = _load()
    data["total_pot"] = amount
    _save(data)


def get_config() -> tuple[dict, str]:
    data = _load()
    config = _normalise_config(data.get("config"))
    legacy_tf = data.get("tf_weights")
    if isinstance(legacy_tf, dict) and (not isinstance(data.get("config"), dict) or "tf_weights" not in data.get("config", {})):
        config["tf_weights"] = _normalise_tf_weights(legacy_tf)
    updated_at = data.get("config_updated_at", "")
    return config, updated_at


def set_config(config_updates: dict) -> str:
    data = _load()
    current = _normalise_config(data.get("config"))
    legacy_tf = data.get("tf_weights")
    if isinstance(legacy_tf, dict) and (not isinstance(data.get("config"), dict) or "tf_weights" not in data.get("config", {})):
        current["tf_weights"] = _normalise_tf_weights(legacy_tf)
    merged = _merge_config(current, config_updates)
    updated_at = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
    data["config"] = merged
    data["config_updated_at"] = updated_at
    data["tf_weights"] = merged["tf_weights"]
    _save(data)
    return updated_at


def get_tf_weights() -> dict:
    config, _ = get_config()
    return config["tf_weights"]


def set_tf_weights(weights: dict):
    set_config({"tf_weights": weights})


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
        data["config"] = _normalise_config(data.get("config"))
        if isinstance(data.get("tf_weights"), dict):
            data["config"]["tf_weights"] = _normalise_tf_weights(data["tf_weights"])
        data["tf_weights"] = data["config"]["tf_weights"]
        if "config_updated_at" not in data:
            data["config_updated_at"] = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
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
