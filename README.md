# FeSTGCN
FeSTGCN: A Frequency-enhanced Spatio-Temporal Graph Convolutional Network for Traffic Flow Prediction under Adaptive Signal Timing

This is just a preliminary version, and we will continue to make changes based on reviewer comments

## 2024.1.12 update：updated Wilcoxon_test

## update 2024.1.15： updated uncertainty analysis
The uncertainty analysis method we use comes from the random deactivating connective weights approach (ARDCW). If you are interested, please check their original article at https://link.springer.com/article/10.1007/s11053-022-10051-w.
Please note that ：
-   since the authors do not provide the source code, we cannot guarantee whether the reproduced results fully reflect the superiority of the original algorithm.
-   Performing ARDCW on FeSTGCN can be very time-consuming, as ARDCW is similar to an ensemble method, and thus the runtime depends on the number of models you adopt

## reviewers' comments
This is the comment of two reviewers on the paper, thank you very much for their patience, we will do our best to make changes!

Reviewer #1: Should make the formatting in a proper way. In page 18, GPU name should be correct. Authors should give compendium about the FeSTGCN layers.

Reviewer #2: 
-        Long Abstract but opaque in term of the used data, time interval, …
-        Only using MAE doesn't guarantee the outperformance of a model. The lack of metric analysis (you only have RMSE, R2 and MAE) is very concrete.
-        Lack of highlighted research gaps, problem statement, pursued goal and motivation should be fulfilled with more in-depth literature and updated works.
-        ARIMA introduced in 2018???
-        The inconsistency of the reference list is clear. arXiv are not peer reviewed. When a work from 2017 is in arXiv and after 7 years no attempts for publication has been done, why it should be a reference?
-        English proofread by a native expert is mandatory. Being consistent in third passive strongly is emphasized. 'our focus', 'we proposed'…
-         Keywords are general. Should show the specificity of the work.
-        Figs like 5 don't have axes title.
-        Conclusion can be justified in a much better form.
-       Follow the APIN guidelines in organizing the draft.
-       Doublecheck that all the used acronyms are defined at the first appearnce.
Technically:
1.        The given contributions are DECLINED. 1, 2 and 3 have the same meaning but different wordings. MUST be reformulated. One of the main problem is that this work doesn't have checked any benchmark data!!!!
2.        This work simply presents a model not algorithm. There is a long difference between model and algorithm. The claim for 'adaptive' as mentioned in keywords neither presented nor satisfied.
3.        If we consider a new algorithm (which is not),
A.        Exactly notify which part/parameter was the subject of improvement/adaptability and why???
B.        Where can the readers find the adaptive procedure???
C.        You must approve that the results don't happen by chance using nonparametric statistical tests like Wilcoxon rank sum test (or any),
D.        It should be ranked among other variants or algorithms through the Freidman test using benchmark datasets in a fair comparison.

4.        All the used definitions come from the own knowledge of authors?? No scholars have used such concepts??

5.        Long discussion on the used datasets. Based on section 3.1 you have used data from Shandong Province, China. Therefore:
A.        The gathered datasets in the field of ML must be validated using benchmarks and documenting the work based on them cannot be certified.
B.        Data validation is the practice of checking the integrity, accuracy and structure of data before it is used for an operation, Range check, Format check, Consistency check.
C.        Whether you are building inference/predictive model, you will achieve better results by first verifying that you have chosen the optimal set of features to train your model with. Therefore, getting better results obviously is expectable.

6.        Since the given model works with traffic data, it needs for continues updating and feeding data. The big concern is to get decisions from the system and database which need to be updated continuously while the given model cannot. Therefore, it cannot always be optimal. In such conditions it can consider only small-scale areas and due to the high computational cost is not compatible with meta data. Such issues make it cumbersome and costly whereas difficulties for adapting to new situations also increases its complexities. What were your solutions?

7.        The data used in this work inherently are time dependent where their analyses due to heterogeneity cannot provide any generalization from a single study. This mathematically means that it suffers from accurately identifying the correct model to represent the data. How did you judge the efficiently dealing with outliers? Used time interval? Number of samplings per frame? …

8.        Based on Fig 2 and specifically attention weight graph, the presented structure in the hidden layers  and then embedding is built on this assumption that all nodes are equally likely to adopt the entity being studied. This means that sensitivity analysis in showing the importance and influence of the used attributes become meaningless. Prediction/model calibration and sensitivity analysis should be carried out through the weight database of trained model when its optimality was confirmed. Clarify where and how the optimal weight database is stored? How it can be recalled?? look at https://journalofbigdata.springeropen.com/articles/10.1186/s40537-021-00515-w, https://iwaponline.com/jh/article/22/3/562/72506/Updating-the-neural-network-sediment-load-models, https://link.springer.com/chapter/10.1007/978-3-030-44584-3_6, ...
9.        In the case of eq 5 and replaced randomly estimated frequency in Fourier transform with MIF the rationality and mathematical proof should be inserted.
10.        Data analysis, for example using matrix chart to show the process of feature selection is mandatory.
11.        Discussion should be supplemented by A) physical interpretation of the accuracy performance, B) limitation of the used method, C) the evidence of fair comparison (How the optimality of given model in Table 4 were confirmed??? Did you adjust them again for the used data?? pretrained using??? …), D) stability approval and practical difficulties, E) the impact of the bias of the used dataset on feature extracting… (of course with citations).
12.        The lack of uncertainty analysis using state-of-the art techniques like https://link.springer.com/article/10.1007/s11053-022-10051-w... Should be taken into account.
