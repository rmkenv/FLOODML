import pytest
import pandas as pd

from floodml.data.usgs import USGSCollector

# We'll monkeypatch nwis.get_info so we can simulate various API return values
import floodml.data.usgs as usgs_module


class DummyValidator:
    def validate_streamflow(self, df):
        return df


@pytest.fixture
def collector():
    col = USGSCollector(site="01438500")
    # prevent DataValidator from running real code
    col.validator = DummyValidator()
    return col


def test_get_site_info_returns_dict_from_dataframe(monkeypatch, collector):
    """Case: nwis.get_info returns a DataFrame with data."""
    dummy_df = pd.DataFrame([{"station_nm": "Test Site", "dec_lat_va": 40.0, "dec_long_va": -75.0}])
    monkeypatch.setattr(usgs_module.nwis, "get_info", lambda **kwargs: dummy_df)

    result = collector.get_site_info()
    assert isinstance(result, dict)
    assert result["station_nm"] == "Test Site"
    assert result["dec_lat_va"] == 40.0


def test_get_site_info_handles_tuple_with_dataframe(monkeypatch, collector):
    """Case: nwis.get_info returns a tuple containing a DataFrame."""
    dummy_df = pd.DataFrame([{"station_nm": "Tuple Site"}])
    dummy_tuple = ("not_a_df", dummy_df, 123)
    monkeypatch.setattr(usgs_module.nwis, "get_info", lambda **kwargs: dummy_tuple)

    result = collector.get_site_info()
    assert result["station_nm"] == "Tuple Site"


def test_get_site_info_tuple_without_dataframe(monkeypatch, collector):
    """Case: nwis.get_info returns a tuple without any DataFrame-like object."""
    dummy_tuple = ("a", "b", 123)
    monkeypatch.setattr(usgs_module.nwis, "get_info", lambda **kwargs: dummy_tuple)

    result = collector.get_site_info()
    assert result == {}  # should gracefully return empty dict


def test_get_site_info_with_empty_dataframe(monkeypatch, collector):
    """Case: nwis.get_info returns an empty DataFrame."""
    empty_df = pd.DataFrame()
    monkeypatch.setattr(usgs_module.nwis, "get_info", lambda **kwargs: empty_df)

    result = collector.get_site_info()
    assert result == {}


def test_get_site_info_exception(monkeypatch, collector):
    """Case: nwis.get_info raises an exception."""
    def raise_exception(**kwargs):
        raise RuntimeError("Simulated API error")
    monkeypatch.setattr(usgs_module.nwis, "get_info", raise_exception)

    result = collector.get_site_info()
    assert result == {}
