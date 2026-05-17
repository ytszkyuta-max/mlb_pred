from pathlib import Path
import pandas as pd
from pybaseball import statcast


def _dated_cache(cache_path: str, start: str, end: str) -> Path:
    """statcast_cache.parquet → statcast_cache_20240320_20240620.parquet"""
    p = Path(cache_path)
    slug = f"{start.replace('-', '')}_{end.replace('-', '')}"
    return p.with_stem(f"{p.stem}_{slug}")


def load(start: str, end: str, cache_path: str) -> pd.DataFrame:
    cache = _dated_cache(cache_path, start, end)
    if cache.exists():
        print(f"キャッシュから読み込み: {cache}")
        df = pd.read_parquet(cache)
        print(f"読み込み件数: {len(df):,} 投球")
        return df

    print(f"Statcastデータ取得中: {start} → {end}")
    df = statcast(start_dt=start, end_dt=end)
    print(f"取得件数: {len(df):,} 投球")

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    print(f"キャッシュ保存: {cache}")
    return df
