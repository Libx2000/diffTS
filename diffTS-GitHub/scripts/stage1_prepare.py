from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from diffts.config import ensure_dirs, load_config, split_years  # noqa: E402
from diffts.io import assert_same_grid, discover_files, open_variable  # noqa: E402


def inspect(cfg: dict, artifacts: Path) -> tuple[dict, dict[int, Path], dict[int, Path]]:
    files = discover_files(cfg)
    years = sorted(files["thetao"])
    t = open_variable(files["thetao"][years[0]], cfg["data"]["temperature_var"], chunks=None)
    s = open_variable(files["so"][years[0]], cfg["data"]["salinity_var"], chunks=None)
    assert_same_grid(t, s)
    expected = int(cfg["data"]["expected_depth_levels"])
    if t.sizes["depth"] != expected:
        raise ValueError(f"预期 {expected} 层，实际 {t.sizes['depth']} 层")
    metadata = {
        "years": years,
        "n_years": len(years),
        "dimensions": {k: int(v) for k, v in t.sizes.items()},
        "depth": t.depth.values.astype(float).tolist(),
        "lat_range": [float(t.lat.min()), float(t.lat.max())],
        "lon_range": [float(t.lon.min()), float(t.lon.max())],
        "first_time": str(pd.Timestamp(t.time.values[0])),
        "last_time_first_file": str(pd.Timestamp(t.time.values[-1])),
        "temperature_files": {str(y): str(p) for y, p in files["thetao"].items()},
        "salinity_files": {str(y): str(p) for y, p in files["so"].items()},
    }
    with (artifacts / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    t.close()
    s.close()
    return metadata, files["thetao"], files["so"]


def all_dates(cfg: dict, files: dict[str, dict[int, Path]]) -> pd.DatetimeIndex:
    dates = []
    for year in sorted(files["thetao"]):
        t = open_variable(files["thetao"][year], cfg["data"]["temperature_var"], chunks=None)
        s = open_variable(files["so"][year], cfg["data"]["salinity_var"], chunks=None)
        assert_same_grid(t, s)
        idx = pd.DatetimeIndex(t.time.values)
        if not idx.is_monotonic_increasing or idx.has_duplicates:
            raise ValueError(f"{year} 时间坐标非单调或有重复")
        dates.extend(idx.tolist())
        t.close()
        s.close()
    result = pd.DatetimeIndex(dates)
    if result.has_duplicates or not result.is_monotonic_increasing:
        raise ValueError("跨年度时间坐标非单调或有重复")
    return result


def build_sample_index(cfg: dict, dates: pd.DatetimeIndex, artifacts: Path) -> None:
    available = set(dates)
    splits = split_years(cfg)
    history = int(cfg["data"]["history_days"])
    rows = []
    for split, years in splits.items():
        year_set = set(years)
        targets = dates[dates.year.astype(int).isin(years)]
        for target in targets:
            for lead in cfg["data"]["lead_days"]:
                input_end = target - pd.Timedelta(days=int(lead))
                history_dates = pd.date_range(input_end - pd.Timedelta(days=history - 1), input_end, freq="D")
                if input_end.year not in year_set or target.year not in year_set:
                    continue
                if all(d in available and d.year in year_set for d in history_dates):
                    rows.append((split, int(lead), input_end, target))
    frame = pd.DataFrame(rows, columns=["split", "lead", "input_end", "target"])
    frame.to_csv(artifacts / "sample_index.csv", index=False)
    print("样本数：")
    print(frame.groupby(["split", "lead"]).size())


def prepare_statistics(cfg: dict, artifacts: Path, files: dict[str, dict[int, Path]]) -> None:
    train_years = split_years(cfg)["train"]
    all_years = sorted(files["thetao"])
    time_chunk = int(cfg["data"].get("time_chunk", 16))
    mode = cfg["stage1"].get("qc_mode", "full")
    sample_n = int(cfg["stage1"].get("sample_days_per_year", 12))

    # 获取坐标和掩膜。
    first_year = train_years[0]
    t0 = open_variable(files["thetao"][first_year], cfg["data"]["temperature_var"], chunks=None)
    s0 = open_variable(files["so"][first_year], cfg["data"]["salinity_var"], chunks=None)
    mask = np.isfinite(t0.isel(time=0).values) & np.isfinite(s0.isel(time=0).values)
    np.save(artifacts / "wet_mask.npy", mask)
    np.savez(artifacts / "coordinates.npz", depth=t0.depth.values, lat=t0.lat.values, lon=t0.lon.values)
    shape = (12, t0.sizes["depth"], t0.sizes["lat"], t0.sizes["lon"])
    t0.close()
    s0.close()

    total_count = np.zeros((2, shape[1]), dtype=np.float64)
    total_sum = np.zeros_like(total_count)
    total_sumsq = np.zeros_like(total_count)
    clim_sum = np.zeros((2, *shape), dtype=np.float64)
    clim_count = np.zeros((2, *shape), dtype=np.float64)
    qc_rows = []

    specs = [("thetao", cfg["data"]["temperature_var"]), ("so", cfg["data"]["salinity_var"])]
    for var_idx, (key, var_name) in enumerate(specs):
        for year in tqdm(all_years, desc=f"质检 {key}"):
            da = open_variable(files[key][year], var_name, chunks={"time": time_chunk})
            work = da
            if mode == "sample" and da.sizes["time"] > sample_n:
                indices = np.linspace(0, da.sizes["time"] - 1, sample_n, dtype=int)
                work = da.isel(time=indices)
            work64 = work.astype("float64")
            reduced = xr.Dataset({
                "count": work64.count(dim=("time", "lat", "lon")),
                "sum": work64.sum(dim=("time", "lat", "lon"), skipna=True),
                "sumsq": (work64 * work64).sum(dim=("time", "lat", "lon"), skipna=True),
                "min": work64.min(dim=("time", "lat", "lon"), skipna=True),
                "max": work64.max(dim=("time", "lat", "lon"), skipna=True),
            }).compute()
            c = reduced["count"].values
            sm = reduced["sum"].values
            ss = reduced["sumsq"].values
            # 标准化参数严格只使用训练年份，避免验证/测试信息泄漏。
            if year in train_years:
                total_count[var_idx] += c
                total_sum[var_idx] += sm
                total_sumsq[var_idx] += ss
            possible = work.sizes["time"] * work.sizes["lat"] * work.sizes["lon"]
            qc_rows.append({
                "variable": key,
                "year": year,
                "n_days_used": int(work.sizes["time"]),
                "valid_fraction": float(c.sum() / (possible * work.sizes["depth"])),
                "minimum": float(reduced["min"].min()),
                "maximum": float(reduced["max"].max()),
                "mean": float(sm.sum() / c.sum()),
                "std": float(np.sqrt(max(ss.sum() / c.sum() - (sm.sum() / c.sum()) ** 2, 0.0))),
            })

            if year in train_years and cfg["stage1"].get("create_monthly_climatology", True) and mode == "full":
                # 一次 groupby 处理全年12个月，避免对同一年度文件重复扫描12次。
                grouped_sum = da.astype("float64").groupby("time.month").sum("time", skipna=True).compute()
                grouped_count = da.groupby("time.month").count("time").compute()
                for month in grouped_sum["month"].values.astype(int):
                    clim_sum[var_idx, month - 1] += grouped_sum.sel(month=month).values
                    clim_count[var_idx, month - 1] += grouped_count.sel(month=month).values
            da.close()

    mean = total_sum / np.maximum(total_count, 1.0)
    variance = total_sumsq / np.maximum(total_count, 1.0) - mean**2
    std = np.sqrt(np.maximum(variance, 1e-12))
    np.savez(artifacts / "normalization.npz", mean=mean.astype(np.float32), std=std.astype(np.float32), count=total_count)
    pd.DataFrame(qc_rows).to_csv(artifacts / "qc_by_year.csv", index=False)

    if cfg["stage1"].get("create_monthly_climatology", True) and mode == "full":
        clim = clim_sum / np.maximum(clim_count, 1.0)
        coords = np.load(artifacts / "coordinates.npz")
        ds = xr.Dataset(
            {
                "thetao": (("month", "depth", "lat", "lon"), clim[0].astype(np.float32)),
                "so": (("month", "depth", "lat", "lon"), clim[1].astype(np.float32)),
            },
            coords={"month": np.arange(1, 13), "depth": coords["depth"], "lat": coords["lat"], "lon": coords["lon"]},
        )
        encoding = {name: {"zlib": True, "complevel": 4, "dtype": "float32"} for name in ds.data_vars}
        ds.to_netcdf(artifacts / "monthly_climatology.nc", encoding=encoding)
        ds.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT / "config.yaml"))
    parser.add_argument("--mode", choices=("inspect", "prepare"), default="prepare")
    args = parser.parse_args()
    cfg = load_config(args.config)
    artifacts, _ = ensure_dirs(cfg)
    _, _, _ = inspect(cfg, artifacts)
    if args.mode == "inspect":
        return
    files = discover_files(cfg)
    dates = all_dates(cfg, files)
    build_sample_index(cfg, dates, artifacts)
    prepare_statistics(cfg, artifacts, files)
    print(f"阶段一完成，输出目录：{artifacts}")


if __name__ == "__main__":
    main()
