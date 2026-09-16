# HyperDense literature notes

Recorded 2026-09-11 following the user's request to research successful applications of hypercomplex dense layers and save the findings.

These are research notes for future experiment selection. They do not change the active internal-replacement experiment, its frozen protocol, queue, or additional $20 allocation. Reported paper results have not been independently reproduced here.

## Main finding

Successful applications use hypercomplex structure for a particular operation or representation. The literature supports selective internal replacement, intentional component grouping, and evaluating parameter efficiency alongside accuracy. It does not establish a universally best dimension among complex (2D), quaternion (4D), and octonion (8D).

Fixed-algebra HyperDense, learned parameterized hypercomplex multiplication (PHM), and complex frequency-domain MLPs are related but distinct methods. Results for one do not automatically establish benefits for the others.

## Relevant primary sources

### Quaternion Transformers — Tay et al., ACL 2019

[Lightweight and Efficient Neural Natural Language Processing with Quaternion Networks](https://aclanthology.org/P19-1145.pdf)

The authors tested a partial conversion of attention projections and a fuller conversion that also changed the position-wise feed-forward layers. On English–Vietnamese translation, the partial version achieved 30.9 BLEU versus 28.4 for the real Transformer, with 29M versus 44M parameters, excluding embeddings. Full conversion was worse on several translation tasks, including English–Romanian.

Implication: test attention projections and feed-forward layers separately before combining replacements. More replaced layers need not improve accuracy. See Tables 2–3 and the architecture description.

### Learned PHM — Zhang et al., ICLR 2021

[Beyond Fully-Connected Layers with Quaternions: Parameterization of Hypercomplex Multiplications with 1/n Parameters](https://arxiv.org/pdf/2102.08597)

PHM learns the matrices defining component interactions instead of fixing Hamilton-product rules. The n=4 PHM Transformer beat the fixed-quaternion Transformer on six of seven translation benchmarks, but did not consistently beat the original real Transformer. Table 3 reports slightly slower training for some compressed models despite faster inference.

The weight parameter count is kd/n + n³, plus output biases. Thus the approximate 1/n saving assumes sufficiently large maps. Calculated for an unshared 32→32 PHM layer including 32 biases: n=2 gives 552 parameters; n=4 gives 352; n=8 gives 672. The learned-rule overhead makes n=8 larger than n=4 here. These are calculated PHM counts, not counts for our fixed-algebra HyperDense layers.

Implication: learned PHM is an informative separate control for whether fixed multiplication rules are too restrictive. Count actual parameters and measure runtime.

### Complex frequency-domain MLPs — Yi et al., NeurIPS 2023

[Frequency-domain MLPs are More Effective Learners in Time Series Forecasting (FreTS)](https://proceedings.nips.cc/paper_files/paper/2023/file/f1d16af76939f476b5f040fd1398c0a3-Paper-Conference.pdf)

FreTS uses complex MLPs after Fourier conversion. The paper also replaces corresponding MLP components in DLinear and NLinear. Section 4.2 reports average DLinear improvements of 6.4% MAE and 11.4% RMSE on Exchange, and 4.9% MAE and 3.5% RMSE on Weather.

Implication: this is a particularly relevant forecasting follow-up. It changes both representation and computation, so compare against real-valued processing of the same Fourier inputs to isolate the contribution of complex structure. It is not evidence that an arbitrary time-domain HyperDense substitution will yield the same gains.

### Octonion frequency-window aggregation — Yakir, Tsur and Permuter, 2025 preprint

[Efficient Time Series Forecasting via Hyper-Complex Models and Frequency Aggregation (FIA-Net)](https://arxiv.org/html/2502.19983v1)

The HC-MLP combines four complex-valued STFT windows as an octonion. The paper reports comparable forecasting accuracy with fewer parameters than its window-mixing alternative; Section 5.2 includes a comparison with both backbones using four windows.

Implication: this offers a concrete interpretation of 8D as four related complex windows. Appendix D.4 also compares quaternion, octonion and sedenion versions, but changes the number of windows with the algebra. Consequently, those comparisons do not isolate algebra dimension from window partitioning. Treat this as a promising design hypothesis, not proof that 8D is best.

### HyperDense on financial series — Kycia and Niemczynowicz, 2024 preprint

[Hypercomplex neural network in time series forecasting of stock data](https://arxiv.org/abs/2401.04632)

The study uses a 4D HyperDense input layer in a small forecasting network with four related financial series. It reports comparable MAE with fewer parameters and sensitivity to input ordering. The searched 4D algebras include quaternions, coquaternions and Cl(1,1).

Its architecture search compares HyperDense-led networks with convolution- and LSTM-led networks. This differs from replacing internal dense maps within fixed DLinear, TSMixer and iTransformer architectures. The preprint describes unequal search-space sizes, so its best-model comparisons do not isolate algebra alone.

### Evidence about meaningful component grouping — Qiu et al., Interspeech 2020

[Quaternion Neural Networks for Multi-Channel Distant Speech Recognition](https://www.isca-archive.org/interspeech_2020/qiu20_interspeech.pdf)

With four microphone signals, the quaternion LSTM achieved 28.7% phoneme error versus 30.2% for a real LSTM on distant TIMIT using FBANK features, averaged over five runs. Copying one microphone into all quaternion components gave approximately equal performance to a single-channel real LSTM: 32.1% versus 32.3%.

This is recurrent-network evidence, not a standalone dense-layer result. It nevertheless motivates testing distinct related components against arbitrary grouping or repeated copies.

## Recommended follow-ups for this project

These are recommendations inferred from the sources, not established findings on our Copper data.

1. Finish the existing balanced internal-replacement comparison. It measures fixed hypercomplex structure at the selected sites without introducing a different representation.
2. Prioritize a frequency-domain DLinear comparison: real processing of Fourier inputs, complex MLP processing, and—if affordable—4D/8D aggregation of related windows. Match observed history, window information, preprocessing, training opportunities and output dimensions across controls.
3. Test component grouping explicitly. Compare a meaningful grouping with frozen permutations; do not select grouping on the evaluation period. The current temporal-coordinate/latent-coordinate grouping differs from the microphone and STFT examples.
4. Compare selected groups of internal layers: attention projections separately from feed-forward maps, with real, 2D, 4D, 8D and relevant compression controls. Validate any Conv1d-equivalent replacement before using it.
5. Consider learned PHM as a separate family, accounting for its learned-rule overhead. This tests whether relaxing fixed algebra rules helps.
6. Report accuracy, parameter count, training time, inference latency and memory separately. Preserve equal tuning opportunities, multiple seeds, chronological splits and the existing low-rank controls; random weight-tying controls would further test whether any benefit is algebra-specific.

The previous Copper periods have already been inspected. Further results on them remain exploratory. Broader claims require independent data. Before launching any follow-up, record its hypothesis, controls, expected cost and stopping criteria within the remaining authorized allocation.
