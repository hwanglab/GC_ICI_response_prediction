**GC ICI treatment response prediction model**

We developed a predictive model for treatment response to immune checkpoint inhibitors (ICIs) in gastric cancer (GC) patients. To overcome the problem of small sample sizes, we implemented the model in Bayesian and ensemble learning frameworks. The model is based on deep Gaussian processes (DGPs) [1,2,3], which are a multilayer extension of Gaussian processes (GPs), which themselves are powerful nonparametric Bayesian methods for nonlinear function estimation problems. 

To test our model, we collected patient samples from our collaborating hospitals (Yonsei and ST Mary) in Korea and from publications [4,5,6]. We collected 135 samples, of which 48 patients responded to the treatment and 87 did not. We specifically split the data in this way: we used the Kwon data, which includes all MSI-H tumors (N=27) but has even numbers of R (13) and NR (14), as the test data to emphasize the value of our method compared to the method that uses patients' MSI-H status as a predictive marker. We obtained an AUC of 0.813 for binary classification (R vs NR) (sensitivity is 0.769 and specificity is 0.714) with our 32 gastric cancer (GC) signatures [7], identified from TCGA mutation data, which have shown statistically significant associations with clinical outcomes in GC patients [8]. We also extended the model to deal with whole genes: gene expressions were converted into images using DeepInsight [9] and a CNN was used as a feature extractor. We saw an improvement in prediction performance compared to the prediction model with the 32 genes. This work will be presented at this year's AACR ANNUAL MEETING in San Diego.

Please check out our AACR poster [https://www.hwanglab.org/aacr2024_dgp]

**Refereces**
1. Damianou, A., Lawrence, N.: Deep Gaussian Processes. Proceedings of the International Conference on Articial Intelligence and Statistics (AISTATS), 2013
2. Cutajar, K et al. Random Feature Expansions for Deep Gaussian Processes, ICML 2017
3. Gia-Lac T et al. Calibrating Deep Convolutional Gaussian Processes, AISTATS 2019
4. Kim ST et al. Comprehensive molecular characterization of clinical responses to PD-1 inhibition in metastatic gastric cancer. Nat Med. 2018 (PMID: 30013197)
5. Chida K et al. Transcriptomic Profiling of MSI-H/dMMR Gastrointestinal Tumors to Identify Determinants of Responsiveness to Anti-PD-1 Therapy. Clin Cancer Res. 2022 (PMID: 35254400)
6. Kwon M et al. Determinants of Response and Intrinsic Resistance to PD-1 Blockade in Microsatellite Instability-High Gastric Cancer. Cancer Discov. 2021  (PMID: 33846173).
7. Park S et al. An integrative somatic mutation analysis to identify pathways linked with survival outcomes across 19 cancer types. Bioinformatics. 2016 (PMID: 26635139)
8. Cheong JH et al,. Development and validation of a prognostic and predictive 32-gene signature for gastric cancer. Nat Commun. 2022 (PMID: 35140202)
9. Sharma, A. et al. DeepInsight: A methodology to transform a non-image data to an image for convolution neural network architecture. Sci Rep 9, 11399 (2019)
