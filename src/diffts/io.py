from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr


DIM_CANDIDATES = {
    "time": ("time", "valid_time", "date"),
    "depth": ("depth", "deptht", "elevation", "lev", "level", "z"),
    "lat": ("latitude", "lat", "nav_lat", "y"),
    "lon": ("longitude", "lon", "nav_lon", "x"),
}


def _find_name(names: Iterable[str], candidates: tuple[str, ...], label: str) -> str:
    names = list(names)
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise KeyError(f"无法识别{label}维度；现有名称：{names}")


def canonicalize(da: xr.DataArray) -> xr.DataArray:
    dims = list(da.dims)
    rename: dict[str, str] = {}
    for canonical, candidates in DIM_CANDIDATES.items():
        original = _find_name(dims, candidates, canonical)
        if original != canonical:
            rename[original] = canonical
    da = da.rename(rename)
    return da.transpose("time", "depth", "lat", "lon")


def open_variable(path: str | Path, variable: str, chunks: dict | None = None) -> xr.DataArray:
    ds = xr.open_dataset(path, chunks=chunks)
    if variable not in ds.data_vars:
        candidates = [k for k, v in ds.data_vars.items() if v.ndim == 4]
        if len(candidates) != 1:
            ds.close()
            raise KeyError(f"{path} 中找不到变量 {variable}；数据变量为 {list(ds.data_vars)}")
        variable = candidates[0]
    return canonicalize(ds[variable])


def discover_files(cfg: dict) -> dict[str, dict[int, Path]]:
    root = Path(cfg["data"]["root"])
    specs = {
        "thetao": (cfg["data"]["temperature_dir"], r"thetao_glor_(\d{4})_0_1\.5deg\.nc$"),
        "so": (cfg["data"]["salinity_dir"], r"so_glor_(\d{4})_0_1\.5deg\.nc$"),
    }
    result: dict[str, dict[int, Path]] = {}
    for key, (folder, pattern) in specs.items():
        files: dict[int, Path] = {}
        for path in (root / folder).glob("*.nc"):
            match = re.search(pattern, path.name)
            if match:
                files[int(match.group(1))] = path
        if not files:
            raise FileNotFoundError(f"{root / folder} 中没有找到年度 NetCDF 文件")
        result[key] = files
    missing_t = sorted(set(result["so"]) - set(result["thetao"]))
    missing_s = sorted(set(result["thetao"]) - set(result["so"]))
    if missing_t or missing_s:
        raise ValueError(f"温盐年份不匹配：缺温度={missing_t}，缺盐度={missing_s}")
    return result


def assert_same_grid(a: xr.DataArray, b: xr.DataArray) -> None:
    for dim in ("depth", "lat", "lon"):
        if a.sizes[dim] != b.sizes[dim] or not np.allclose(a[dim].values, b[dim].values):
            raise ValueError(f"温度和盐度的 {dim} 坐标不一致")
    ta = pd.DatetimeIndex(a.time.values)
    tb = pd.DatetimeIndex(b.time.values)
    if not ta.equals(tb):
        raise ValueError("温度和盐度的时间坐标不一致")


class YearFileStore:
    """按年惰性打开 NetCDF；适合31年数据，避免一次性拼接所有文件。"""

    def __init__(self, cfg: dict, max_open_years: int | None = None):
        self.cfg = cfg
        self.files = discover_files(cfg)
        self.max_open_years = max_open_years or int(cfg["data"].get("max_open_years", 3))
        self._cache: OrderedDict[tuple[str, int], xr.DataArray] = OrderedDict()

    def _var_name(self, key: str) -> str:
        return self.cfg["data"]["temperature_var"] if key == "thetao" else self.cfg["data"]["salinity_var"]

    def get(self, key: str, year: int) -> xr.DataArray:
        cache_key = (key, int(year))
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]
        da = open_variable(self.files[key][int(year)], self._var_name(key), chunks=None)
        self._cache[cache_key] = da
        while len(self._cache) > 2 * self.max_open_years:
            _, old = self._cache.popitem(last=False)
            old.close()
        return da

    def read_state(self, dates: Iterable[pd.Timestamp]) -> np.ndarray:
        """返回 [time, variable(2), depth, lat, lon]，变量顺序为 T、S。"""
        outputs = []
        for date in [pd.Timestamp(d) for d in dates]:
            fields = []
            for key in ("thetao", "so"):
                da = self.get(key, date.year)
                try:
                    field = da.sel(time=np.datetime64(date)).values
                except KeyError as exc:
                    raise KeyError(f"{key} 缺少日期 {date.date()}") from exc
                fields.append(np.asarray(field, dtype=np.float32))
            outputs.append(np.stack(fields, axis=0))
        return np.stack(outputs, axis=0)

    def close(self) -> None:
        for da in self._cache.values():
            da.close()
        self._cache.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state
