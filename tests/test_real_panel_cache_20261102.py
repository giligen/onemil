"""Pass 4: the cached research-panel window equals the inline read the parity tests used and cannot leak mutations."""
import os

import pandas as pd
import pytest

from tests.real_panel_cache import PANEL, real_panel_window


@pytest.mark.skipif(not os.path.exists(PANEL), reason='research panel not present')
def test_cached_window_equals_the_inline_parquet_read_and_is_isolated_per_caller():
    import pyarrow.parquet as pq
    reb = pd.Timestamp('2021-02-08')
    lo = reb - pd.Timedelta(days=460)
    tbl = pq.read_table(PANEL, columns=['symbol', 'bar_date', 'close', 'volume'],
                        filters=[('bar_date', '>=', lo.to_pydatetime()), ('bar_date', '<', reb.to_pydatetime())])
    inline = tbl.to_pandas()
    inline['close'] = inline['close'].astype('float32')
    first = real_panel_window(reb)
    pd.testing.assert_frame_equal(first, inline)
    first['close'] = 0.0                                     # a mutating caller...
    second = real_panel_window('2021-02-08')                 # ...cannot reach the next caller
    pd.testing.assert_frame_equal(second, inline)
    assert str(second['close'].dtype) == 'float32'
