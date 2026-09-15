import numpy as np
import pandas as pd
from typing import Callable, Literal
from workshop_infrastructure.datasets.helio import HelioNetCDFDataset


class SolarWindDSDataset(HelioNetCDFDataset):
    """
    Child class of HelioNetCDFDataset showing how to build a downstream dataset from an
    OMNI solar-wind index aligned to the Surya index.

    All ``HelioNetCDFDataset`` keyword arguments (``index_path``, ``scalers``, ``channels``,
    ``s3_cache_dir``, etc.) are accepted via ``**kwargs`` and forwarded to the base class.
    ``load_forecast_frames`` defaults to ``False`` here (solar wind forecasting supplies its
    own labels, so future Surya frames are never fetched); pass it explicitly to override.

    Additional Args:
        return_surya_stack: If True (default), include the Surya image stack in the returned dict.
            Set to False to return only the solar-wind label (useful for label inspection).
        max_number_of_samples: Cap the dataset length at this value. Useful for quick experiments.
        label_transform: Optional callable applied to the ``ds_target_column`` column of the
            index to produce the ``normalized_target`` label. Signature:
            ``(series: pd.Series) -> pd.Series``.  If ``None``, the raw values are used as-is.
            Define this at the call site (e.g., in ``build_datasets()``) to keep normalization
            logic out of the dataset class.
        ds_index_path: Path to the downstream solar-wind CSV index (already split into
            train/val — pass the file matching whichever Surya split this dataset wraps).
        ds_time_column: Column name in the index to use as the event timestamp (typically
            "Estimated_source_time" — the back-propagated solar departure time).
        ds_target_column: Column name in the index to regress on (e.g. "Speed, km/s").
        ds_fill_value: OMNI fill-value sentinel used by ``ds_target_column`` for missing
            measurements (e.g. 99999.9 for speed). Rows equal to this value are dropped
            before alignment. Pass ``None`` to skip filtering.
        ds_time_tolerance: Maximum allowed time offset when matching Surya and DS indices
            (e.g., ``"6min"``). Unmatched entries are dropped.
        ds_match_direction: Merge direction passed to ``pd.merge_asof``. Use ``"nearest"``
            when the DS timestamp is already a point estimate rather than an event start.

    Raises:
        ValueError: If ``ds_index_path`` is not provided, or if no overlap exists
            between the Surya and DS indices within the specified tolerance.
    """

    def __init__(
        self,
        # Downstream-specific parameters
        return_surya_stack: bool = True,
        max_number_of_samples: int | None = None,
        label_transform: Callable[[pd.Series], pd.Series] | None = None,
        ds_index_path: str | None = None,
        ds_time_column: str | None = None,
        ds_target_column: str = "Speed, km/s",
        ds_fill_value: float | None = 99999.9,
        ds_time_tolerance: str | None = None,
        ds_match_direction: Literal["forward", "backward", "nearest"] = "nearest",
        # All HelioNetCDFDataset parameters (index_path, scalers, channels, s3_*, etc.)
        **kwargs,
    ):
        if ds_match_direction not in ["forward", "backward", "nearest"]:
            raise ValueError("ds_match_direction must be one of 'forward', 'backward', or 'nearest'")

        # load_forecast_frames defaults to False here: solar wind forecasting supplies its
        # own labels, so future Surya frames never need to be fetched from disk/S3.
        kwargs.setdefault("load_forecast_frames", False)
        super().__init__(**kwargs)

        self.return_surya_stack = return_surya_stack

        # Load ds index and find intersection with Surya index
        if ds_index_path is not None:
            self.ds_index = pd.read_csv(ds_index_path)
        else:
            raise ValueError("ds_index_path must be provided for SolarWindDSDataset")

        # OMNI marks missing measurements with a fixed sentinel rather than NaN (e.g.
        # 99999.9 for speed) — drop those rows before they contaminate the target.
        if ds_fill_value is not None:
            self.ds_index = self.ds_index.loc[
                self.ds_index[ds_target_column] != ds_fill_value, :
            ]

        self.ds_index["ds_index"] = pd.to_datetime(
            self.ds_index[ds_time_column]
        ).values.astype("datetime64[ns]")
        self.ds_index.sort_values("ds_index", inplace=True)

        # Apply label transform if provided; otherwise use raw target values.
        if label_transform is not None:
            self.ds_index["normalized_target"] = label_transform(self.ds_index[ds_target_column])
        else:
            self.ds_index["normalized_target"] = self.ds_index[ds_target_column]

        # Create Surya valid indices and find closest match to DS index
        self.df_valid_indices = pd.DataFrame(
            {"valid_indices": self.valid_indices}
        ).sort_values("valid_indices")
        self.df_valid_indices = pd.merge_asof(
            self.df_valid_indices,
            self.ds_index,
            right_on="ds_index",
            left_on="valid_indices",
            direction=ds_match_direction,
        )
        # Remove duplicates keeping closest match
        self.df_valid_indices["index_delta"] = np.abs(
            self.df_valid_indices["valid_indices"] - self.df_valid_indices["ds_index"]
        )
        self.df_valid_indices = self.df_valid_indices.sort_values(
            ["ds_index", "index_delta"]
        )
        self.df_valid_indices.drop_duplicates(
            subset="ds_index", keep="first", inplace=True
        )
        # Enforce a maximum time tolerance for matches
        if ds_time_tolerance is not None:
            self.df_valid_indices = self.df_valid_indices.loc[
                self.df_valid_indices["index_delta"] <= pd.Timedelta(ds_time_tolerance),
                :,
            ]
            if len(self.df_valid_indices) == 0:
                raise ValueError("No intersection between Surya and DS indices")

        # Override valid indices variables to reflect matches between Surya and DS
        self.valid_indices = [
            pd.Timestamp(date) for date in self.df_valid_indices["valid_indices"]
        ]
        self.adjusted_length = len(self.valid_indices)
        self.df_valid_indices.set_index("valid_indices", inplace=True)

        if max_number_of_samples is not None and max_number_of_samples < self.adjusted_length:
            self.valid_indices = self.valid_indices[:max_number_of_samples]
            self.df_valid_indices = self.df_valid_indices.iloc[:max_number_of_samples]
            self.adjusted_length = max_number_of_samples

    def __len__(self):
        return self.adjusted_length

    def __getitem__(self, idx: int) -> dict:
        """
        Args:
            idx: Dataset index.

        Returns:
            Dictionary containing:
                forecast (np.float32): Normalized solar-wind target label.
                ds_index (str): ISO-format timestamp from the solar-wind index.
            When ``return_surya_stack=True``, also includes all keys from
            ``HelioNetCDFDataset.__getitem__`` (ts, time_delta_input, lead_time_delta, etc.).
        """
        sample = super().__getitem__(idx=idx) if self.return_surya_stack else {}
        sample["forecast"] = self.df_valid_indices.iloc[idx]["normalized_target"].astype(np.float32)
        sample["ds_index"] = self.df_valid_indices["ds_index"].iloc[idx].isoformat()
        return sample
