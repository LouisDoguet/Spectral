# OPNO-capacity final ranking (promoted)

12 trained / 15 configs. Ranked by cases stable (of 4) then mean NN relative-L2 density error vs MUSCL (lower = better). Errors are per case; blow-up counts as 1.0.

| rank | name | axis | P | n_elem | muscl_cells | rollout | kernel | w_osc | w_alpha | width | depth | opno_hidden | opno_channels | fusion_hidden | n_stable | beats_dg | mean_nn_err | err_sod | err_lax | err_shu | err_random |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | W128_D3_OH128_OC4_FH48 | opno_search_full | 4 | 32 | 64 | 512 | 5 | 1e-05 | 1e-05 | 128 | 3 | 128 | 4 | 48 | 4 | 4 | 0.04382 | 0.00485 | 0.04904 | 0.05737 | 0.06402 |
| 2 | W48_D1_OH32_OC16_FH48 | opno_search_full | 4 | 32 | 64 | 512 | 5 | 1e-05 | 1e-05 | 48 | 1 | 32 | 16 | 48 | 4 | 4 | 0.04384 | 0.004379 | 0.04599 | 0.06053 | 0.06446 |
| 3 | W16_D1_OH16_OC4_FH96 | opno_search_full | 4 | 32 | 64 | 512 | 5 | 1e-05 | 1e-05 | 16 | 1 | 16 | 4 | 96 | 4 | 4 | 0.04403 | 0.003464 | 0.05169 | 0.05751 | 0.06345 |
| 4 | W128_D3_OH128_OC4_FH48_P6 | opno_search_full | 6 | 32 | 64 | 512 | 7 | 1e-05 | 1e-05 | 128 | 3 | 128 | 4 | 48 | 4 | 4 | 0.04471 | 0.004532 | 0.04822 | 0.04431 | 0.08177 |
| 5 | W64_D1_OH64_OC64_FH64_P6 | opno_search_full | 6 | 32 | 64 | 512 | 7 | 1e-05 | 1e-05 | 64 | 1 | 64 | 64 | 64 | 4 | 4 | 0.04516 | 0.003188 | 0.04708 | 0.04499 | 0.08539 |
| 6 | W64_D1_OH64_OC64_FH64 | opno_search_full | 4 | 32 | 64 | 512 | 5 | 1e-05 | 1e-05 | 64 | 1 | 64 | 64 | 64 | 4 | 4 | 0.04545 | 0.004893 | 0.05385 | 0.05856 | 0.0645 |
| 7 | W16_D1_OH16_OC4_FH96_P6 | opno_search_full | 6 | 32 | 64 | 512 | 7 | 1e-05 | 1e-05 | 16 | 1 | 16 | 4 | 96 | 4 | 4 | 0.04597 | 0.004627 | 0.04921 | 0.04685 | 0.0832 |
| 8 | W48_D1_OH32_OC16_FH48_P6 | opno_search_full | 6 | 32 | 64 | 512 | 7 | 1e-05 | 1e-05 | 48 | 1 | 32 | 16 | 48 | 4 | 4 | 0.04674 | 0.003712 | 0.04923 | 0.0509 | 0.0831 |
| 9 | W16_D3_OH64_OC64_FH32 | opno_search_full | 4 | 32 | 64 | 512 | 5 | 1e-05 | 1e-05 | 16 | 3 | 64 | 64 | 32 | 4 | 3 | 0.04699 | 0.005869 | 0.05628 | 0.06158 | 0.06424 |
| 10 | W16_D3_OH64_OC64_FH32_P6 | opno_search_full | 6 | 32 | 64 | 512 | 7 | 1e-05 | 1e-05 | 16 | 3 | 64 | 64 | 32 | 4 | 3 | 0.04787 | 0.005917 | 0.05394 | 0.04838 | 0.08324 |
| 11 | W64_D1_OH64_OC64_FH64_P8 | opno_search_full | 8 | 32 | 64 | 512 | 9 | 1e-05 | 1e-05 | 64 | 1 | 64 | 64 | 64 | 4 | 4 | 0.06019 | 0.003225 | 0.0446 | 0.03965 | 0.1533 |
| 12 | W16_D3_OH64_OC64_FH32_P8 | opno_search_full | 8 | 32 | 64 | 512 | 9 | 1e-05 | 1e-05 | 16 | 3 | 64 | 64 | 32 | 4 | 4 | 0.07012 | 0.004409 | 0.05615 | 0.04901 | 0.1709 |

**Did not train:** W128_D3_OH128_OC4_FH48_P8, W16_D1_OH16_OC4_FH96_P8, W48_D1_OH32_OC16_FH48_P8
