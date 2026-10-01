# Public data attribution and measurement limits

`odmr_public.dat` is distributed unchanged from LiuLiu (2025),
"20241224-1651-50_ODMR_data_ch0_raw.dat", Figshare.

- Source: https://figshare.com/articles/dataset/28788437
- DOI: https://doi.org/10.6084/m9.figshare.28788437
- Download: https://ndownloader.figshare.com/files/53646563
- License: Creative Commons Attribution 4.0 International,
  https://creativecommons.org/licenses/by/4.0/
- SHA-256: 748e54dcd66b64ba18999f230b752dfb770d951536102c86fa40bd8160e71ac7
- Size: 18,974,276 bytes.

Derived plots and audit statistics were computed by this project. They do not imply endorsement by the data author. The file contains 4,693 sweeps with 311 frequency points (2740--3050 MHz, 1 MHz spacing). Its header records an analog acquisition channel, /Dev1/AI0. The floating-point records and counts/s label do not establish independent integer Poisson counts. Per-sweep field truth is unavailable: this dataset is not used to report magnetic-field RMSE.

The raw file is downloaded on demand by `download_data.py` and excluded from Git. Published audit statistics and plots are derived from the unchanged file identified above.
