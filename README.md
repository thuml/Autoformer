# Autoformer (NeurIPS 2021)

Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting

Time series forecasting is a critical demand for real applications. Enlighted by the classic time series analysis and stochastic process theory, we propose the Autoformer as a general series forecasting model [[paper](https://arxiv.org/abs/2106.13008)]. **Autoformer goes beyond the Transformer family and achieves the series-wise connection for the first time.**

In long-term forecasting, Autoformer achieves SOTA, with a **38% relative improvement** on six benchmarks, covering five practical applications: **energy, traffic, economics, weather and disease**.

:triangular_flag_on_post:**News** (2022.02-2022.03) Autoformer has been deployed in [2022 Winter Olympics](https://en.wikipedia.org/wiki/2022_Winter_Olympics) to provide weather forecasting for competition venues, including wind speed and temperature.

## Autoformer vs. Transformers

**1. Deep decomposition architecture**

We renovate the Transformer as a deep decomposition architecture, which can progressively decompose the trend and seasonal components during the forecasting process.

<p align="center">
<img src=".\pic\Autoformer.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overall architecture of Autoformer.
</p>

**2. Series-wise Auto-Correlation mechanism**

Inspired by the stochastic process theory, we design the Auto-Correlation mechanism, which can discover period-based dependencies and aggregate the information at the series level. This empowers the model with inherent log-linear complexity. This series-wise connection contrasts clearly from the previous self-attention family.

<p align="center">
<img src=".\pic\Auto-Correlation.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 2.</b> Auto-Correlation mechansim.
</p>

## Get Started

1. Install Python 3.6, PyTorch 1.9.0.
2. Download data. You can obtain all the six benchmarks from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/e1ccfff39ad541908bae/) or [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing). **All the datasets are well pre-processed** and can be used easily.
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results by:

```bash
bash ./scripts/ETT_script/Autoformer_ETTm1.sh
bash ./scripts/ECL_script/Autoformer.sh
bash ./scripts/Exchange_script/Autoformer.sh
bash ./scripts/Traffic_script/Autoformer.sh
bash ./scripts/Weather_script/Autoformer.sh
bash ./scripts/ILI_script/Autoformer.sh
```

4. Sepcial-designed implementation

- **Speedup Auto-Correlation:** We built the Auto-Correlation mechanism as a batch-normalization-style block to make it more memory-access friendly. See the [paper](https://arxiv.org/abs/2106.13008) for details.

- **Without the position embedding:** Since the series-wise connection will inherently keep the sequential information, Autoformer does not need the position embedding, which is different from Transformers.

### Reproduce with Docker

To easily reproduce the results using Docker, conda and Make,  you can follow the next steps:
1. Initialize the docker image using: `make init`. 
2. Download the datasets using: `make get_dataset`.
3. Run each script in `scripts/` using `make run_module module="bash scripts/ETT_script/Autoformer_ETTm1.sh"` for each script.
4. Alternatively, run all the scripts at once:
```
for file in `ls scripts`; do make run_module module="bash scripts/$script"; done
```

## Main Results

We experiment on six benchmarks, covering five main-stream applications. We compare our model with ten baselines, including Informer, N-BEATS, etc. Generally, for the long-term forecasting setting, Autoformer achieves SOTA, with a **38% relative improvement** over previous baselines.

<p align="center">
<img src=".\pic\results.png" height = "550" alt="" align=center />
</p>

## Baselines

We will keep adding series forecasting models to expand this repo:

- [x] Autoformer
- [x] Informer
- [x] Transformer
- [ ] LogTrans
- [ ] Reformer
- [ ] N-BEATS

## Citation

If you find this repo useful, please cite our paper. 

```
@inproceedings{wu2021autoformer,
  title={Autoformer: Decomposition Transformers with {Auto-Correlation} for Long-Term Series Forecasting},
  author={Haixu Wu and Jiehui Xu and Jianmin Wang and Mingsheng Long},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

## Contact

If you have any question or want to use the code, please contact whx20@mails.tsinghua.edu.cn .

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/zhouhaoyi/Informer2020

https://github.com/zhouhaoyi/ETDataset

https://github.com/laiguokun/multivariate-time-series-data

