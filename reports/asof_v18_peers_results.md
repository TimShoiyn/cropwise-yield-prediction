# v18 peers results

Walk-forward by year on `productivity_estimate_peers.csv`.

## Best ML vs best naive baseline

| variant   | scenario       |   asof_tag | asof_label   | model                         | model_type   |     n |     r2 |   rmse |    mae |    mape | best_naive   |   naive_mape |   naive_mae |   naive_rmse |   naive_r2 |
|:----------|:---------------|-----------:|:-------------|:------------------------------|:-------------|------:|-------:|-------:|-------:|--------:|:-------------|-------------:|------------:|-------------:|-----------:|
| plus      | all_crops      |      07_01 | 1 Jul        | catboost_mae_l20_residual     | ml           | 26029 | 0.7498 | 0.6533 | 0.4035 | 17.2724 | crop_mean    |      25.1672 |      0.5727 |       0.8362 |     0.5901 |
| plus      | all_crops      |      08_01 | 1 Aug        | catboost_mae_l20              | ml           | 26029 | 0.3875 | 1.0222 | 0.3827 | 15.2453 | crop_mean    |      25.1672 |      0.5727 |       0.8362 |     0.5901 |
| plus      | all_crops      |      09_01 | 1 Sep        | catboost_mae_l20              | ml           | 26029 | 0.4162 | 0.9979 | 0.3772 | 15.1255 | crop_mean    |      25.1672 |      0.5727 |       0.8362 |     0.5901 |
| plus      | sunflower      |      07_01 | 1 Jul        | catboost_mae_l20_residual     | ml           | 13416 | 0.4009 | 0.423  | 0.3409 | 16.299  | global_mean  |      22.4044 |      0.4462 |       0.5497 |    -0.0117 |
| plus      | sunflower      |      08_01 | 1 Aug        | catboost_mae_l20              | ml           | 13416 | 0.5624 | 0.3615 | 0.2881 | 13.6096 | global_mean  |      22.4044 |      0.4462 |       0.5497 |    -0.0117 |
| plus      | sunflower      |      09_01 | 1 Sep        | catboost_fast_l10             | ml           | 13416 | 0.5656 | 0.3602 | 0.2875 | 13.7036 | global_mean  |      22.4044 |      0.4462 |       0.5497 |    -0.0117 |
| plus      | wheat_combined |      07_01 | 1 Jul        | catboost_fast_l10_residual    | ml           |  7144 | 0.6532 | 0.5648 | 0.4484 | 16.5081 | crop_mean    |      29.3382 |      0.7666 |       0.9362 |     0.0471 |
| plus      | wheat_combined |      08_01 | 1 Aug        | catboost_mae_l20_residual     | ml           |  7144 | 0.6929 | 0.5315 | 0.4221 | 15.3389 | crop_mean    |      29.3382 |      0.7666 |       0.9362 |     0.0471 |
| plus      | wheat_combined |      09_01 | 1 Sep        | catboost_mae_l20_residual     | ml           |  7144 | 0.697  | 0.5279 | 0.4201 | 15.2632 | crop_mean    |      29.3382 |      0.7666 |       0.9362 |     0.0471 |
| strict    | all_crops      |      07_01 | 1 Jul        | catboost_mae_l20_residual     | ml           | 26029 | 0.7489 | 0.6544 | 0.4057 | 17.3222 | crop_mean    |      25.1672 |      0.5727 |       0.8362 |     0.5901 |
| strict    | all_crops      |      08_01 | 1 Aug        | catboost_mae_l20              | ml           | 26029 | 0.3991 | 1.0124 | 0.3806 | 15.2217 | crop_mean    |      25.1672 |      0.5727 |       0.8362 |     0.5901 |
| strict    | all_crops      |      09_01 | 1 Sep        | catboost_mae_l20              | ml           | 26029 | 0.4013 | 1.0106 | 0.3786 | 15.1651 | crop_mean    |      25.1672 |      0.5727 |       0.8362 |     0.5901 |
| strict    | sunflower      |      07_01 | 1 Jul        | catboost_mae_l20_residual     | ml           | 13416 | 0.3866 | 0.428  | 0.345  | 16.3695 | global_mean  |      22.4044 |      0.4462 |       0.5497 |    -0.0117 |
| strict    | sunflower      |      08_01 | 1 Aug        | catboost_mae_l20              | ml           | 13416 | 0.5653 | 0.3604 | 0.2873 | 13.5496 | global_mean  |      22.4044 |      0.4462 |       0.5497 |    -0.0117 |
| strict    | sunflower      |      09_01 | 1 Sep        | catboost_mae_l20              | ml           | 13416 | 0.5652 | 0.3604 | 0.287  | 13.5911 | global_mean  |      22.4044 |      0.4462 |       0.5497 |    -0.0117 |
| strict    | wheat_combined |      07_01 | 1 Jul        | catboost_fast_l10_residual    | ml           |  7144 | 0.6556 | 0.5628 | 0.4463 | 16.3631 | crop_mean    |      29.3382 |      0.7666 |       0.9362 |     0.0471 |
| strict    | wheat_combined |      08_01 | 1 Aug        | catboost_shallow_l30_residual | ml           |  7144 | 0.6945 | 0.5301 | 0.4207 | 15.3901 | crop_mean    |      29.3382 |      0.7666 |       0.9362 |     0.0471 |
| strict    | wheat_combined |      09_01 | 1 Sep        | catboost_mae_l20              | ml           |  7144 | 0.7065 | 0.5196 | 0.4126 | 15.0296 | crop_mean    |      29.3382 |      0.7666 |       0.9362 |     0.0471 |

## Full grid

| scenario       |   asof_tag | asof_label   | variant   | model                         | model_type   |     n |      r2 |   rmse |    mae |    mape |
|:---------------|-----------:|:-------------|:----------|:------------------------------|:-------------|------:|--------:|-------:|-------:|--------:|
| sunflower      |      07_01 | 1 Jul        | strict    | global_mean                   | naive        | 13416 | -0.0117 | 0.5497 | 0.4462 | 22.4044 |
| sunflower      |      07_01 | 1 Jul        | strict    | crop_mean                     | naive        | 13416 | -0.0117 | 0.5497 | 0.4462 | 22.4044 |
| sunflower      |      07_01 | 1 Jul        | strict    | crop_year_trend               | naive        | 13416 | -0.0136 | 0.5503 | 0.4466 | 22.6668 |
| sunflower      |      07_01 | 1 Jul        | strict    | catboost_fast_l10             | ml           | 13416 |  0.3876 | 0.4277 | 0.3449 | 16.4483 |
| sunflower      |      07_01 | 1 Jul        | strict    | catboost_fast_l10_residual    | ml           | 13416 |  0.3876 | 0.4277 | 0.3449 | 16.4483 |
| sunflower      |      07_01 | 1 Jul        | strict    | catboost_shallow_l30          | ml           | 13416 |  0.3901 | 0.4268 | 0.3453 | 16.496  |
| sunflower      |      07_01 | 1 Jul        | strict    | catboost_shallow_l30_residual | ml           | 13416 |  0.3901 | 0.4268 | 0.3453 | 16.496  |
| sunflower      |      07_01 | 1 Jul        | strict    | catboost_mae_l20              | ml           | 13416 |  0.3864 | 0.4281 | 0.3451 | 16.393  |
| sunflower      |      07_01 | 1 Jul        | strict    | catboost_mae_l20_residual     | ml           | 13416 |  0.3866 | 0.428  | 0.345  | 16.3695 |
| sunflower      |      08_01 | 1 Aug        | strict    | global_mean                   | naive        | 13416 | -0.0117 | 0.5497 | 0.4462 | 22.4044 |
| sunflower      |      08_01 | 1 Aug        | strict    | crop_mean                     | naive        | 13416 | -0.0117 | 0.5497 | 0.4462 | 22.4044 |
| sunflower      |      08_01 | 1 Aug        | strict    | crop_year_trend               | naive        | 13416 | -0.0136 | 0.5503 | 0.4466 | 22.6668 |
| sunflower      |      08_01 | 1 Aug        | strict    | catboost_fast_l10             | ml           | 13416 |  0.5668 | 0.3597 | 0.2871 | 13.5898 |
| sunflower      |      08_01 | 1 Aug        | strict    | catboost_fast_l10_residual    | ml           | 13416 |  0.5669 | 0.3597 | 0.287  | 13.5902 |
| sunflower      |      08_01 | 1 Aug        | strict    | catboost_shallow_l30          | ml           | 13416 |  0.5673 | 0.3595 | 0.2871 | 13.6038 |
| sunflower      |      08_01 | 1 Aug        | strict    | catboost_shallow_l30_residual | ml           | 13416 |  0.5673 | 0.3595 | 0.2871 | 13.6038 |
| sunflower      |      08_01 | 1 Aug        | strict    | catboost_mae_l20              | ml           | 13416 |  0.5653 | 0.3604 | 0.2873 | 13.5496 |
| sunflower      |      08_01 | 1 Aug        | strict    | catboost_mae_l20_residual     | ml           | 13416 |  0.5638 | 0.361  | 0.2878 | 13.5742 |
| sunflower      |      09_01 | 1 Sep        | strict    | global_mean                   | naive        | 13416 | -0.0117 | 0.5497 | 0.4462 | 22.4044 |
| sunflower      |      09_01 | 1 Sep        | strict    | crop_mean                     | naive        | 13416 | -0.0117 | 0.5497 | 0.4462 | 22.4044 |
| sunflower      |      09_01 | 1 Sep        | strict    | crop_year_trend               | naive        | 13416 | -0.0136 | 0.5503 | 0.4466 | 22.6668 |
| sunflower      |      09_01 | 1 Sep        | strict    | catboost_fast_l10             | ml           | 13416 |  0.5693 | 0.3587 | 0.2859 | 13.6108 |
| sunflower      |      09_01 | 1 Sep        | strict    | catboost_fast_l10_residual    | ml           | 13416 |  0.5693 | 0.3587 | 0.2859 | 13.6108 |
| sunflower      |      09_01 | 1 Sep        | strict    | catboost_shallow_l30          | ml           | 13416 |  0.5694 | 0.3586 | 0.286  | 13.6236 |
| sunflower      |      09_01 | 1 Sep        | strict    | catboost_shallow_l30_residual | ml           | 13416 |  0.5694 | 0.3586 | 0.286  | 13.6236 |
| sunflower      |      09_01 | 1 Sep        | strict    | catboost_mae_l20              | ml           | 13416 |  0.5652 | 0.3604 | 0.287  | 13.5911 |
| sunflower      |      09_01 | 1 Sep        | strict    | catboost_mae_l20_residual     | ml           | 13416 |  0.5651 | 0.3604 | 0.2871 | 13.6051 |
| wheat_combined |      07_01 | 1 Jul        | strict    | global_mean                   | naive        |  7144 | -0.0337 | 0.9751 | 0.797  | 31.4902 |
| wheat_combined |      07_01 | 1 Jul        | strict    | crop_mean                     | naive        |  7144 |  0.0471 | 0.9362 | 0.7666 | 29.3382 |
| wheat_combined |      07_01 | 1 Jul        | strict    | crop_year_trend               | naive        |  7144 |  0.0283 | 0.9454 | 0.7741 | 29.6329 |
| wheat_combined |      07_01 | 1 Jul        | strict    | catboost_fast_l10             | ml           |  7144 |  0.654  | 0.5642 | 0.4461 | 16.3812 |
| wheat_combined |      07_01 | 1 Jul        | strict    | catboost_fast_l10_residual    | ml           |  7144 |  0.6556 | 0.5628 | 0.4463 | 16.3631 |
| wheat_combined |      07_01 | 1 Jul        | strict    | catboost_shallow_l30          | ml           |  7144 |  0.6526 | 0.5653 | 0.4477 | 16.5048 |
| wheat_combined |      07_01 | 1 Jul        | strict    | catboost_shallow_l30_residual | ml           |  7144 |  0.6545 | 0.5637 | 0.4468 | 16.4323 |
| wheat_combined |      07_01 | 1 Jul        | strict    | catboost_mae_l20              | ml           |  7144 |  0.6432 | 0.5729 | 0.4525 | 16.5441 |
| wheat_combined |      07_01 | 1 Jul        | strict    | catboost_mae_l20_residual     | ml           |  7144 |  0.6457 | 0.5709 | 0.4514 | 16.4412 |
| wheat_combined |      08_01 | 1 Aug        | strict    | global_mean                   | naive        |  7144 | -0.0337 | 0.9751 | 0.797  | 31.4902 |
| wheat_combined |      08_01 | 1 Aug        | strict    | crop_mean                     | naive        |  7144 |  0.0471 | 0.9362 | 0.7666 | 29.3382 |
| wheat_combined |      08_01 | 1 Aug        | strict    | crop_year_trend               | naive        |  7144 |  0.0283 | 0.9454 | 0.7741 | 29.6329 |
| wheat_combined |      08_01 | 1 Aug        | strict    | catboost_fast_l10             | ml           |  7144 |  0.6923 | 0.532  | 0.4214 | 15.4506 |
| wheat_combined |      08_01 | 1 Aug        | strict    | catboost_fast_l10_residual    | ml           |  7144 |  0.695  | 0.5297 | 0.4203 | 15.4081 |
| wheat_combined |      08_01 | 1 Aug        | strict    | catboost_shallow_l30          | ml           |  7144 |  0.693  | 0.5314 | 0.4219 | 15.5135 |
| wheat_combined |      08_01 | 1 Aug        | strict    | catboost_shallow_l30_residual | ml           |  7144 |  0.6945 | 0.5301 | 0.4207 | 15.3901 |
| wheat_combined |      08_01 | 1 Aug        | strict    | catboost_mae_l20              | ml           |  7144 |  0.6893 | 0.5346 | 0.4231 | 15.415  |
| wheat_combined |      08_01 | 1 Aug        | strict    | catboost_mae_l20_residual     | ml           |  7144 |  0.688  | 0.5357 | 0.4248 | 15.4218 |
| wheat_combined |      09_01 | 1 Sep        | strict    | global_mean                   | naive        |  7144 | -0.0337 | 0.9751 | 0.797  | 31.4902 |
| wheat_combined |      09_01 | 1 Sep        | strict    | crop_mean                     | naive        |  7144 |  0.0471 | 0.9362 | 0.7666 | 29.3382 |
| wheat_combined |      09_01 | 1 Sep        | strict    | crop_year_trend               | naive        |  7144 |  0.0283 | 0.9454 | 0.7741 | 29.6329 |
| wheat_combined |      09_01 | 1 Sep        | strict    | catboost_fast_l10             | ml           |  7144 |  0.7095 | 0.5169 | 0.411  | 15.053  |
| wheat_combined |      09_01 | 1 Sep        | strict    | catboost_fast_l10_residual    | ml           |  7144 |  0.7121 | 0.5146 | 0.4097 | 15.0909 |
| wheat_combined |      09_01 | 1 Sep        | strict    | catboost_shallow_l30          | ml           |  7144 |  0.71   | 0.5165 | 0.4107 | 15.1515 |
| wheat_combined |      09_01 | 1 Sep        | strict    | catboost_shallow_l30_residual | ml           |  7144 |  0.7109 | 0.5157 | 0.4104 | 15.1087 |
| wheat_combined |      09_01 | 1 Sep        | strict    | catboost_mae_l20              | ml           |  7144 |  0.7065 | 0.5196 | 0.4126 | 15.0296 |
| wheat_combined |      09_01 | 1 Sep        | strict    | catboost_mae_l20_residual     | ml           |  7144 |  0.7007 | 0.5247 | 0.4172 | 15.1175 |
| all_crops      |      07_01 | 1 Jul        | strict    | global_mean                   | naive        | 26029 | -0.0027 | 1.3078 | 0.6947 | 30.6875 |
| all_crops      |      07_01 | 1 Jul        | strict    | crop_mean                     | naive        | 26029 |  0.5901 | 0.8362 | 0.5727 | 25.1672 |
| all_crops      |      07_01 | 1 Jul        | strict    | crop_year_trend               | naive        | 26029 |  0.5798 | 0.8466 | 0.5767 | 25.4399 |
| all_crops      |      07_01 | 1 Jul        | strict    | catboost_fast_l10             | ml           | 26029 |  0.5523 | 0.8739 | 0.4299 | 18.4677 |
| all_crops      |      07_01 | 1 Jul        | strict    | catboost_fast_l10_residual    | ml           | 26029 |  0.7519 | 0.6505 | 0.404  | 17.4153 |
| all_crops      |      07_01 | 1 Jul        | strict    | catboost_shallow_l30          | ml           | 26029 |  0.5163 | 0.9083 | 0.4418 | 19.0869 |
| all_crops      |      07_01 | 1 Jul        | strict    | catboost_shallow_l30_residual | ml           | 26029 |  0.7482 | 0.6554 | 0.4089 | 17.6516 |
| all_crops      |      07_01 | 1 Jul        | strict    | catboost_mae_l20              | ml           | 26029 |  0.3666 | 1.0394 | 0.4257 | 17.3804 |
| all_crops      |      07_01 | 1 Jul        | strict    | catboost_mae_l20_residual     | ml           | 26029 |  0.7489 | 0.6544 | 0.4057 | 17.3222 |
| all_crops      |      08_01 | 1 Aug        | strict    | global_mean                   | naive        | 26029 | -0.0027 | 1.3078 | 0.6947 | 30.6875 |
| all_crops      |      08_01 | 1 Aug        | strict    | crop_mean                     | naive        | 26029 |  0.5901 | 0.8362 | 0.5727 | 25.1672 |
| all_crops      |      08_01 | 1 Aug        | strict    | crop_year_trend               | naive        | 26029 |  0.5798 | 0.8466 | 0.5767 | 25.4399 |
| all_crops      |      08_01 | 1 Aug        | strict    | catboost_fast_l10             | ml           | 26029 |  0.5649 | 0.8615 | 0.3902 | 16.4977 |
| all_crops      |      08_01 | 1 Aug        | strict    | catboost_fast_l10_residual    | ml           | 26029 |  0.781  | 0.6112 | 0.3612 | 15.3852 |
| all_crops      |      08_01 | 1 Aug        | strict    | catboost_shallow_l30          | ml           | 26029 |  0.5478 | 0.8782 | 0.4027 | 17.2178 |
| all_crops      |      08_01 | 1 Aug        | strict    | catboost_shallow_l30_residual | ml           | 26029 |  0.7774 | 0.6161 | 0.3661 | 15.6205 |
| all_crops      |      08_01 | 1 Aug        | strict    | catboost_mae_l20              | ml           | 26029 |  0.3991 | 1.0124 | 0.3806 | 15.2217 |
| all_crops      |      08_01 | 1 Aug        | strict    | catboost_mae_l20_residual     | ml           | 26029 |  0.7765 | 0.6174 | 0.3635 | 15.3118 |
| all_crops      |      09_01 | 1 Sep        | strict    | global_mean                   | naive        | 26029 | -0.0027 | 1.3078 | 0.6947 | 30.6875 |
| all_crops      |      09_01 | 1 Sep        | strict    | crop_mean                     | naive        | 26029 |  0.5901 | 0.8362 | 0.5727 | 25.1672 |
| all_crops      |      09_01 | 1 Sep        | strict    | crop_year_trend               | naive        | 26029 |  0.5798 | 0.8466 | 0.5767 | 25.4399 |
| all_crops      |      09_01 | 1 Sep        | strict    | catboost_fast_l10             | ml           | 26029 |  0.565  | 0.8614 | 0.3836 | 16.2955 |
| all_crops      |      09_01 | 1 Sep        | strict    | catboost_fast_l10_residual    | ml           | 26029 |  0.7826 | 0.609  | 0.3577 | 15.3241 |
| all_crops      |      09_01 | 1 Sep        | strict    | catboost_shallow_l30          | ml           | 26029 |  0.5302 | 0.8952 | 0.4028 | 17.2746 |
| all_crops      |      09_01 | 1 Sep        | strict    | catboost_shallow_l30_residual | ml           | 26029 |  0.7793 | 0.6136 | 0.3616 | 15.4653 |
| all_crops      |      09_01 | 1 Sep        | strict    | catboost_mae_l20              | ml           | 26029 |  0.4013 | 1.0106 | 0.3786 | 15.1651 |
| all_crops      |      09_01 | 1 Sep        | strict    | catboost_mae_l20_residual     | ml           | 26029 |  0.7777 | 0.6158 | 0.3599 | 15.1799 |
| sunflower      |      07_01 | 1 Jul        | plus      | global_mean                   | naive        | 13416 | -0.0117 | 0.5497 | 0.4462 | 22.4044 |
| sunflower      |      07_01 | 1 Jul        | plus      | crop_mean                     | naive        | 13416 | -0.0117 | 0.5497 | 0.4462 | 22.4044 |
| sunflower      |      07_01 | 1 Jul        | plus      | crop_year_trend               | naive        | 13416 | -0.0136 | 0.5503 | 0.4466 | 22.6668 |
| sunflower      |      07_01 | 1 Jul        | plus      | catboost_fast_l10             | ml           | 13416 |  0.4041 | 0.4219 | 0.3408 | 16.411  |
| sunflower      |      07_01 | 1 Jul        | plus      | catboost_fast_l10_residual    | ml           | 13416 |  0.4041 | 0.4219 | 0.3408 | 16.411  |
| sunflower      |      07_01 | 1 Jul        | plus      | catboost_shallow_l30          | ml           | 13416 |  0.4077 | 0.4206 | 0.3403 | 16.3948 |
| sunflower      |      07_01 | 1 Jul        | plus      | catboost_shallow_l30_residual | ml           | 13416 |  0.4075 | 0.4207 | 0.3404 | 16.39   |
| sunflower      |      07_01 | 1 Jul        | plus      | catboost_mae_l20              | ml           | 13416 |  0.4025 | 0.4225 | 0.3411 | 16.3051 |
| sunflower      |      07_01 | 1 Jul        | plus      | catboost_mae_l20_residual     | ml           | 13416 |  0.4009 | 0.423  | 0.3409 | 16.299  |
| sunflower      |      08_01 | 1 Aug        | plus      | global_mean                   | naive        | 13416 | -0.0117 | 0.5497 | 0.4462 | 22.4044 |
| sunflower      |      08_01 | 1 Aug        | plus      | crop_mean                     | naive        | 13416 | -0.0117 | 0.5497 | 0.4462 | 22.4044 |
| sunflower      |      08_01 | 1 Aug        | plus      | crop_year_trend               | naive        | 13416 | -0.0136 | 0.5503 | 0.4466 | 22.6668 |
| sunflower      |      08_01 | 1 Aug        | plus      | catboost_fast_l10             | ml           | 13416 |  0.5646 | 0.3606 | 0.2886 | 13.7025 |
| sunflower      |      08_01 | 1 Aug        | plus      | catboost_fast_l10_residual    | ml           | 13416 |  0.5646 | 0.3606 | 0.2886 | 13.7025 |
| sunflower      |      08_01 | 1 Aug        | plus      | catboost_shallow_l30          | ml           | 13416 |  0.5676 | 0.3594 | 0.2877 | 13.6614 |
| sunflower      |      08_01 | 1 Aug        | plus      | catboost_shallow_l30_residual | ml           | 13416 |  0.5675 | 0.3594 | 0.2877 | 13.6601 |
| sunflower      |      08_01 | 1 Aug        | plus      | catboost_mae_l20              | ml           | 13416 |  0.5624 | 0.3615 | 0.2881 | 13.6096 |
| sunflower      |      08_01 | 1 Aug        | plus      | catboost_mae_l20_residual     | ml           | 13416 |  0.5619 | 0.3618 | 0.2884 | 13.6146 |
| sunflower      |      09_01 | 1 Sep        | plus      | global_mean                   | naive        | 13416 | -0.0117 | 0.5497 | 0.4462 | 22.4044 |
| sunflower      |      09_01 | 1 Sep        | plus      | crop_mean                     | naive        | 13416 | -0.0117 | 0.5497 | 0.4462 | 22.4044 |
| sunflower      |      09_01 | 1 Sep        | plus      | crop_year_trend               | naive        | 13416 | -0.0136 | 0.5503 | 0.4466 | 22.6668 |
| sunflower      |      09_01 | 1 Sep        | plus      | catboost_fast_l10             | ml           | 13416 |  0.5656 | 0.3602 | 0.2875 | 13.7036 |
| sunflower      |      09_01 | 1 Sep        | plus      | catboost_fast_l10_residual    | ml           | 13416 |  0.5656 | 0.3602 | 0.2875 | 13.7036 |
| sunflower      |      09_01 | 1 Sep        | plus      | catboost_shallow_l30          | ml           | 13416 |  0.5622 | 0.3616 | 0.289  | 13.788  |
| sunflower      |      09_01 | 1 Sep        | plus      | catboost_shallow_l30_residual | ml           | 13416 |  0.5621 | 0.3617 | 0.2891 | 13.7892 |
| sunflower      |      09_01 | 1 Sep        | plus      | catboost_mae_l20              | ml           | 13416 |  0.558  | 0.3634 | 0.2895 | 13.7501 |
| sunflower      |      09_01 | 1 Sep        | plus      | catboost_mae_l20_residual     | ml           | 13416 |  0.5595 | 0.3627 | 0.2893 | 13.739  |
| wheat_combined |      07_01 | 1 Jul        | plus      | global_mean                   | naive        |  7144 | -0.0337 | 0.9751 | 0.797  | 31.4902 |
| wheat_combined |      07_01 | 1 Jul        | plus      | crop_mean                     | naive        |  7144 |  0.0471 | 0.9362 | 0.7666 | 29.3382 |
| wheat_combined |      07_01 | 1 Jul        | plus      | crop_year_trend               | naive        |  7144 |  0.0283 | 0.9454 | 0.7741 | 29.6329 |
| wheat_combined |      07_01 | 1 Jul        | plus      | catboost_fast_l10             | ml           |  7144 |  0.6497 | 0.5676 | 0.4505 | 16.6282 |
| wheat_combined |      07_01 | 1 Jul        | plus      | catboost_fast_l10_residual    | ml           |  7144 |  0.6532 | 0.5648 | 0.4484 | 16.5081 |
| wheat_combined |      07_01 | 1 Jul        | plus      | catboost_shallow_l30          | ml           |  7144 |  0.6523 | 0.5656 | 0.4486 | 16.6065 |
| wheat_combined |      07_01 | 1 Jul        | plus      | catboost_shallow_l30_residual | ml           |  7144 |  0.6542 | 0.564  | 0.4484 | 16.5609 |
| wheat_combined |      07_01 | 1 Jul        | plus      | catboost_mae_l20              | ml           |  7144 |  0.6434 | 0.5727 | 0.4523 | 16.5457 |
| wheat_combined |      07_01 | 1 Jul        | plus      | catboost_mae_l20_residual     | ml           |  7144 |  0.6487 | 0.5684 | 0.4509 | 16.5239 |
| wheat_combined |      08_01 | 1 Aug        | plus      | global_mean                   | naive        |  7144 | -0.0337 | 0.9751 | 0.797  | 31.4902 |
| wheat_combined |      08_01 | 1 Aug        | plus      | crop_mean                     | naive        |  7144 |  0.0471 | 0.9362 | 0.7666 | 29.3382 |
| wheat_combined |      08_01 | 1 Aug        | plus      | crop_year_trend               | naive        |  7144 |  0.0283 | 0.9454 | 0.7741 | 29.6329 |
| wheat_combined |      08_01 | 1 Aug        | plus      | catboost_fast_l10             | ml           |  7144 |  0.6899 | 0.5341 | 0.4252 | 15.6732 |
| wheat_combined |      08_01 | 1 Aug        | plus      | catboost_fast_l10_residual    | ml           |  7144 |  0.6896 | 0.5344 | 0.425  | 15.5984 |
| wheat_combined |      08_01 | 1 Aug        | plus      | catboost_shallow_l30          | ml           |  7144 |  0.692  | 0.5322 | 0.4227 | 15.6357 |
| wheat_combined |      08_01 | 1 Aug        | plus      | catboost_shallow_l30_residual | ml           |  7144 |  0.6936 | 0.5309 | 0.422  | 15.4774 |
| wheat_combined |      08_01 | 1 Aug        | plus      | catboost_mae_l20              | ml           |  7144 |  0.6862 | 0.5372 | 0.4269 | 15.6271 |
| wheat_combined |      08_01 | 1 Aug        | plus      | catboost_mae_l20_residual     | ml           |  7144 |  0.6929 | 0.5315 | 0.4221 | 15.3389 |
| wheat_combined |      09_01 | 1 Sep        | plus      | global_mean                   | naive        |  7144 | -0.0337 | 0.9751 | 0.797  | 31.4902 |
| wheat_combined |      09_01 | 1 Sep        | plus      | crop_mean                     | naive        |  7144 |  0.0471 | 0.9362 | 0.7666 | 29.3382 |
| wheat_combined |      09_01 | 1 Sep        | plus      | crop_year_trend               | naive        |  7144 |  0.0283 | 0.9454 | 0.7741 | 29.6329 |
| wheat_combined |      09_01 | 1 Sep        | plus      | catboost_fast_l10             | ml           |  7144 |  0.707  | 0.5192 | 0.4144 | 15.3158 |
| wheat_combined |      09_01 | 1 Sep        | plus      | catboost_fast_l10_residual    | ml           |  7144 |  0.7034 | 0.5224 | 0.4172 | 15.439  |
| wheat_combined |      09_01 | 1 Sep        | plus      | catboost_shallow_l30          | ml           |  7144 |  0.705  | 0.5209 | 0.415  | 15.3657 |
| wheat_combined |      09_01 | 1 Sep        | plus      | catboost_shallow_l30_residual | ml           |  7144 |  0.7063 | 0.5197 | 0.4147 | 15.2839 |
| wheat_combined |      09_01 | 1 Sep        | plus      | catboost_mae_l20              | ml           |  7144 |  0.6955 | 0.5292 | 0.4205 | 15.379  |
| wheat_combined |      09_01 | 1 Sep        | plus      | catboost_mae_l20_residual     | ml           |  7144 |  0.697  | 0.5279 | 0.4201 | 15.2632 |
| all_crops      |      07_01 | 1 Jul        | plus      | global_mean                   | naive        | 26029 | -0.0027 | 1.3078 | 0.6947 | 30.6875 |
| all_crops      |      07_01 | 1 Jul        | plus      | crop_mean                     | naive        | 26029 |  0.5901 | 0.8362 | 0.5727 | 25.1672 |
| all_crops      |      07_01 | 1 Jul        | plus      | crop_year_trend               | naive        | 26029 |  0.5798 | 0.8466 | 0.5767 | 25.4399 |
| all_crops      |      07_01 | 1 Jul        | plus      | catboost_fast_l10             | ml           | 26029 |  0.5316 | 0.8938 | 0.4323 | 18.4981 |
| all_crops      |      07_01 | 1 Jul        | plus      | catboost_fast_l10_residual    | ml           | 26029 |  0.7532 | 0.6488 | 0.4024 | 17.3866 |
| all_crops      |      07_01 | 1 Jul        | plus      | catboost_shallow_l30          | ml           | 26029 |  0.5201 | 0.9048 | 0.439  | 18.8156 |
| all_crops      |      07_01 | 1 Jul        | plus      | catboost_shallow_l30_residual | ml           | 26029 |  0.7503 | 0.6526 | 0.4056 | 17.5594 |
| all_crops      |      07_01 | 1 Jul        | plus      | catboost_mae_l20              | ml           | 26029 |  0.368  | 1.0382 | 0.4252 | 17.4048 |
| all_crops      |      07_01 | 1 Jul        | plus      | catboost_mae_l20_residual     | ml           | 26029 |  0.7498 | 0.6533 | 0.4035 | 17.2724 |
| all_crops      |      08_01 | 1 Aug        | plus      | global_mean                   | naive        | 26029 | -0.0027 | 1.3078 | 0.6947 | 30.6875 |
| all_crops      |      08_01 | 1 Aug        | plus      | crop_mean                     | naive        | 26029 |  0.5901 | 0.8362 | 0.5727 | 25.1672 |
| all_crops      |      08_01 | 1 Aug        | plus      | crop_year_trend               | naive        | 26029 |  0.5798 | 0.8466 | 0.5767 | 25.4399 |
| all_crops      |      08_01 | 1 Aug        | plus      | catboost_fast_l10             | ml           | 26029 |  0.5799 | 0.8465 | 0.3881 | 16.2433 |
| all_crops      |      08_01 | 1 Aug        | plus      | catboost_fast_l10_residual    | ml           | 26029 |  0.7796 | 0.6131 | 0.3618 | 15.4271 |
| all_crops      |      08_01 | 1 Aug        | plus      | catboost_shallow_l30          | ml           | 26029 |  0.5613 | 0.865  | 0.3989 | 16.7979 |
| all_crops      |      08_01 | 1 Aug        | plus      | catboost_shallow_l30_residual | ml           | 26029 |  0.7769 | 0.6169 | 0.3662 | 15.6392 |
| all_crops      |      08_01 | 1 Aug        | plus      | catboost_mae_l20              | ml           | 26029 |  0.3875 | 1.0222 | 0.3827 | 15.2453 |
| all_crops      |      08_01 | 1 Aug        | plus      | catboost_mae_l20_residual     | ml           | 26029 |  0.7748 | 0.6198 | 0.3649 | 15.3643 |
| all_crops      |      09_01 | 1 Sep        | plus      | global_mean                   | naive        | 26029 | -0.0027 | 1.3078 | 0.6947 | 30.6875 |
| all_crops      |      09_01 | 1 Sep        | plus      | crop_mean                     | naive        | 26029 |  0.5901 | 0.8362 | 0.5727 | 25.1672 |
| all_crops      |      09_01 | 1 Sep        | plus      | crop_year_trend               | naive        | 26029 |  0.5798 | 0.8466 | 0.5767 | 25.4399 |
| all_crops      |      09_01 | 1 Sep        | plus      | catboost_fast_l10             | ml           | 26029 |  0.5752 | 0.8512 | 0.385  | 16.0737 |
| all_crops      |      09_01 | 1 Sep        | plus      | catboost_fast_l10_residual    | ml           | 26029 |  0.7822 | 0.6095 | 0.3575 | 15.3077 |
| all_crops      |      09_01 | 1 Sep        | plus      | catboost_shallow_l30          | ml           | 26029 |  0.5649 | 0.8615 | 0.3982 | 16.7789 |
| all_crops      |      09_01 | 1 Sep        | plus      | catboost_shallow_l30_residual | ml           | 26029 |  0.778  | 0.6153 | 0.3632 | 15.5675 |
| all_crops      |      09_01 | 1 Sep        | plus      | catboost_mae_l20              | ml           | 26029 |  0.4162 | 0.9979 | 0.3772 | 15.1255 |
| all_crops      |      09_01 | 1 Sep        | plus      | catboost_mae_l20_residual     | ml           | 26029 |  0.7776 | 0.616  | 0.3604 | 15.2306 |
