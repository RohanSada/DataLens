from functools import lru_cache

from app.core.deps import get_settings
from app.services.datalens import DataLens


@lru_cache
def get_datalens() -> DataLens:
    datalens = DataLens(get_settings())
    try:
        datalens.text_to_sql.warmup()
    except Exception:
        pass
    return datalens
