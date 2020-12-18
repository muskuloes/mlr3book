## Tuning with Hyperband {#hyperband}


Besides the more traditional tuning methods, the ecosystem around [mlr3](https://mlr3.mlr-org.com) offers another procedure for hyperparameter optimization called Hyperband implemented in the [mlr3hyperband](https://mlr3hyperband.mlr-org.com) package.
Hyperband is a budget-oriented procedure, weeding out suboptimal performing configurations early on during a partially sequential training process, increasing tuning efficiency as a consequence.
For this, a combination of incremental resource allocation and early stopping is used: As optimization progresses, computational resources are increased for more promising configurations, while less promising ones are terminated early.
To give an introductory analogy, imagine two horse trainers are given eight untrained horses.
Both trainers want to win the upcoming race, but they are only given 32 units of food.
Given that each horse can be fed up to 8 units food ("maximum budget" per horse), there is not enough food for all the horses.
It is critical to identify the most promising horses early, and give them enough food to improve.
So, the trainers need to develop a strategy to split up the food in the best possible way.
The first trainer is very optimistic and wants to explore the full capabilities of a horse, because he does not want to pass a judgment on a horse's performance unless it has been fully trained.
So, he divides his budget by the maximum amount he can give to a horse (lets say eight, so $32 / 8 = 4$) and randomly picks four horses - his budget simply is not enough to fully train more.
Those four horses are then trained to their full capabilities, while the rest is set free.
This way, the trainer is confident about choosing the best out of the four trained horses, but he might have overlooked the horse with the highest potential since he only focused on half of them.
The other trainer is more creative and develops a different strategy.
He thinks, if a horse is not performing well at the beginning, it will also not improve after further training.
Based on this assumption, he decides to give one unit of food to each horse and observes how they develop.
After the initial food is consumed, he checks their performance and kicks the slowest half out of his training regime.
Then, he increases the available food for the remaining, further trains them until the food is consumed again, only to kick out the worst half once more.
He repeats this until the one remaining horse gets the rest of the food.
This means only one horse is fully trained, but on the flip side, he was able to start training with all eight horses.
On race day, all the horses are put on the starting line.
But which trainer will have the winning horse?
The one, who tried to train a maximum amount of horses to their fullest?
Or the other one, who made assumptions about the training progress of his horses?
How the training phases may possibly look like is visualized in figure \@ref(fig:03-optimization-hyperband-001).

<div class="figure" style="text-align: center">
<img src="images/horse_training1.png" alt="Visulization of how the training processes may look like. The left plot corresponds to the non-selective trainer, while the right one to the selective trainer." width="99%" />
<p class="caption">(\#fig:03-optimization-hyperband-001)Visulization of how the training processes may look like. The left plot corresponds to the non-selective trainer, while the right one to the selective trainer.</p>
</div>

Hyperband works very similar in some ways, but also different in others.
It is not embodied by one of the trainers in our analogy, but more by the person, who would pay them.
Hyperband consists of several brackets, each bracket corresponding to a trainer, and we do not care about horses but about hyperparameter configurations of a machine learning algorithm.
The budget is not in terms of food, but in terms of a hyperparameter of the learner that scales in some way with the computational effort.
An example is the number of epochs we train a neural network, or the number of iterations in boosting.
Furthermore, there are not only two brackets (or trainers), but several, each placed at a unique spot between fully explorative of later training stages and extremely selective, equal to higher exploration of early training stages.
The level of selection aggressiveness is handled by a user-defined parameter called $\eta$.
So, $1/\eta$ is the fraction of remaining configurations after a bracket removes his worst performing ones, but $\eta$ is also the factor by that the budget is increased for the next stage.
Because there is a different maximum budget per configuration that makes sense in different scenarios, the user also has to set this as the $R$ parameter.
No further parameters are required for Hyperband -- the full required budget across all brackets is indirectly given by $$(\lfloor \log_{\eta}{R} \rfloor + 1)^2 * R$$ [@Li2016].
To give an idea how a full bracket layout might look like for a specific $R$ and $\eta$, a quick overview is given in the following table.


<table class="kable_wrapper">
<caption>(\#tab:03-optimization-hyperband-002)Hyperband layout for $\eta = 2$ and $R = 8$, consisting of four brackets with $n$ as the amount of active configurations.</caption>
<tbody>
  <tr>
   <td> 

| stage| budget|  n|
|-----:|------:|--:|
|     1|      1|  8|
|     2|      2|  4|
|     3|      4|  2|
|     4|      8|  1|

 </td>
   <td> 

| stage| budget|  n|
|-----:|------:|--:|
|     1|      2|  6|
|     2|      4|  3|
|     3|      8|  1|

 </td>
   <td> 

| stage| budget|  n|
|-----:|------:|--:|
|     1|      4|  4|
|     2|      8|  2|

 </td>
   <td> 

| stage| budget|  n|
|-----:|------:|--:|
|     1|      8|  4|

 </td>
  </tr>
</tbody>
</table>

Of course, early termination based on a performance criterion may be disadvantageous if it is done too aggressively in certain scenarios.
A learner to jumping radically in its estimated performance during the training phase may get the best configurations canceled too early, simply because they do not improve quickly enough compared to others.
In other words, it is often unclear beforehand if having an high amount of configurations $n$, that gets aggressively discarded early, is better than having a high budget $B$ per configuration.
The arising tradeoff, that has to be made, is called the "$n$ versus $B/n$ problem".
To create a balance between selection based on early training performance versus exploration of training performances in later training stages, $\lfloor \log_{\eta}{R} \rfloor + 1$ brackets are constructed with an associated set of varying sized configurations.
Thus, some brackets contain more configurations, with a small initial budget.
In these, a lot are discarded after having been trained for only a short amount of time, corresponding to the selective trainer in our horse analogy.
Others are constructed with fewer configurations, where discarding only takes place after a significant amount of budget was consumed.
The last bracket usually never discards anything, but also starts with only very few configurations -- this is equivalent to the trainer explorative of later stages.
The former corresponds high $n$, while the latter high $B/n$.
Even though different brackets are initialized with a different amount of configurations and different initial budget sizes, each bracket is assigned (approximately) the same budget $(\lfloor \log_{\eta}{R} \rfloor + 1) * R$.

The configurations at the start of each bracket are initialized by random, often uniform sampling.
Note that currently all configurations are trained completely from the beginning, so no online updates of models from stage to stage is happening.

To identify the budget for evaluating Hyperband, the user has to specify explicitly which hyperparameter of the learner influences the budget by extending a single hyperparameter in the [`ParamSet`](https://paradox.mlr-org.com/reference/ParamSet.html) with an argument (`tags = "budget"`), like in the following snippet:


```r
library(paradox)

# Hyperparameter subset of XGBoost
search_space = ParamSet$new(list(
  ParamInt$new("nrounds", lower = 1, upper = 16, tags = "budget"),
  ParamFct$new("booster", levels = c("gbtree", "gblinear", "dart"))
))
```

Thanks to the broad ecosystem of the [mlr3verse](https://mlr3verse.mlr-org.com) a learner does not require a natural budget parameter.
A typical case of this would be decision trees.
By using subsampling as preprocessing with [mlr3pipelines](https://mlr3pipelines.mlr-org.com), we can work around a lacking budget parameter.


```r
library(mlr3tuning)
library(mlr3hyperband)
library(mlr3pipelines)
set.seed(123)

# extend "classif.rpart" with "subsampling" as preprocessing step
ll = po("subsample") %>>% lrn("classif.rpart")

# extend hyperparameters of "classif.rpart" with subsampling fraction as budget
search_space = ParamSet$new(list(
  ParamDbl$new("classif.rpart.cp", lower = 0.001, upper = 0.1),
  ParamInt$new("classif.rpart.minsplit", lower = 1, upper = 10),
  ParamDbl$new("subsample.frac", lower = 0.1, upper = 1, tags = "budget")
))
```

We can now plug the new learner with the extended hyperparameter set into a [`TuningInstanceSingleCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceSingleCrit.html) the same way as usual.
Naturally, Hyperband terminates once all of its brackets are evaluated, so a [`Terminator`](https://bbotk.mlr-org.com/reference/Terminator.html) in the tuning instance acts as an upper bound and should be only set to a low value if one is unsure of how long Hyperband will take to finish under the given settings.


```r
instance = TuningInstanceSingleCrit$new(
  task = tsk("iris"),
  learner = ll,
  resampling = rsmp("holdout"),
  measure = msr("classif.ce"),
  terminator = trm("none"), # hyperband terminates itself
  search_space = search_space
)
```

Now, we initialize a new instance of the [`mlr3hyperband::TunerHyperband`](https://www.rdocumentation.org/packages/mlr3hyperband/topics/TunerHyperband) class and start tuning with it.


```r
tuner = tnr("hyperband", eta = 3)
tuner$optimize(instance)
```

```
## INFO  [15:43:55.913] Starting to optimize 3 parameter(s) with '<TunerHyperband>' and '<TerminatorNone>' 
## INFO  [15:43:55.975] Amount of brackets to be evaluated = 3,  
## INFO  [15:43:55.990] Start evaluation of bracket 1 
## INFO  [15:43:55.997] Training 9 configs with budget of 0.111111 for each 
## INFO  [15:43:56.003] Evaluating 9 configuration(s) 
## INFO  [15:43:58.411] Result of batch 1: 
## INFO  [15:43:58.416]  classif.rpart.cp classif.rpart.minsplit subsample.frac bracket bracket_stage 
## INFO  [15:43:58.416]           0.02533                      3         0.1111       2             0 
## INFO  [15:43:58.416]           0.07348                      5         0.1111       2             0 
## INFO  [15:43:58.416]           0.08490                      3         0.1111       2             0 
## INFO  [15:43:58.416]           0.05026                      6         0.1111       2             0 
## INFO  [15:43:58.416]           0.03940                      4         0.1111       2             0 
## INFO  [15:43:58.416]           0.02540                      7         0.1111       2             0 
## INFO  [15:43:58.416]           0.01200                      4         0.1111       2             0 
## INFO  [15:43:58.416]           0.03961                      4         0.1111       2             0 
## INFO  [15:43:58.416]           0.05762                      6         0.1111       2             0 
## INFO  [15:43:58.416]  budget_scaled budget_real n_configs classif.ce 
## INFO  [15:43:58.416]          1.111      0.1111         9       0.06 
## INFO  [15:43:58.416]          1.111      0.1111         9       0.08 
## INFO  [15:43:58.416]          1.111      0.1111         9       0.02 
## INFO  [15:43:58.416]          1.111      0.1111         9       0.02 
## INFO  [15:43:58.416]          1.111      0.1111         9       0.08 
## INFO  [15:43:58.416]          1.111      0.1111         9       0.02 
## INFO  [15:43:58.416]          1.111      0.1111         9       0.10 
## INFO  [15:43:58.416]          1.111      0.1111         9       0.14 
## INFO  [15:43:58.416]          1.111      0.1111         9       0.04 
## INFO  [15:43:58.416]                                 uhash 
## INFO  [15:43:58.416]  1ccce4fb-8ea2-4ba6-80a7-c327d22255d2 
## INFO  [15:43:58.416]  e2c0b222-f340-4a9d-b648-0007f7d22efe 
## INFO  [15:43:58.416]  03c59bbf-5a8f-4ffc-87c6-9a8cbea10458 
## INFO  [15:43:58.416]  5d96a8c0-a4d9-45ee-9519-df1cd1edd391 
## INFO  [15:43:58.416]  35ae4d87-fb06-4d4b-92ea-adb60ec67e50 
## INFO  [15:43:58.416]  980b9441-dde3-4751-a7d4-084f75d3b7bd 
## INFO  [15:43:58.416]  17e9dd73-8370-4595-9d19-b5898e40e160 
## INFO  [15:43:58.416]  2ae38f46-511c-40bc-8501-9126359d9cf2 
## INFO  [15:43:58.416]  9b66a2cc-b4e2-4c36-b0e5-6b9c11121253 
## INFO  [15:43:58.418] Training 3 configs with budget of 0.333333 for each 
## INFO  [15:43:58.422] Evaluating 3 configuration(s) 
## INFO  [15:43:59.211] Result of batch 2: 
## INFO  [15:43:59.214]  classif.rpart.cp classif.rpart.minsplit subsample.frac bracket bracket_stage 
## INFO  [15:43:59.214]           0.08490                      3         0.3333       2             1 
## INFO  [15:43:59.214]           0.05026                      6         0.3333       2             1 
## INFO  [15:43:59.214]           0.02540                      7         0.3333       2             1 
## INFO  [15:43:59.214]  budget_scaled budget_real n_configs classif.ce 
## INFO  [15:43:59.214]          3.333      0.3333         3       0.02 
## INFO  [15:43:59.214]          3.333      0.3333         3       0.04 
## INFO  [15:43:59.214]          3.333      0.3333         3       0.06 
## INFO  [15:43:59.214]                                 uhash 
## INFO  [15:43:59.214]  fb13cb3a-a04c-45ae-a9de-9cce5a97a1e9 
## INFO  [15:43:59.214]  46bb2c7b-6603-4ecd-a3b3-ea7cf05d5925 
## INFO  [15:43:59.214]  1bfb8771-6407-47da-bff0-e8a6c3e6f101 
## INFO  [15:43:59.216] Training 1 configs with budget of 1 for each 
## INFO  [15:43:59.221] Evaluating 1 configuration(s) 
## INFO  [15:43:59.533] Result of batch 3: 
## INFO  [15:43:59.537]  classif.rpart.cp classif.rpart.minsplit subsample.frac bracket bracket_stage 
## INFO  [15:43:59.537]            0.0849                      3              1       2             2 
## INFO  [15:43:59.537]  budget_scaled budget_real n_configs classif.ce 
## INFO  [15:43:59.537]             10           1         1       0.04 
## INFO  [15:43:59.537]                                 uhash 
## INFO  [15:43:59.537]  a5547edb-c1ee-4fa5-95eb-9157fa7a1580 
## INFO  [15:43:59.538] Start evaluation of bracket 2 
## INFO  [15:43:59.543] Training 5 configs with budget of 0.333333 for each 
## INFO  [15:43:59.546] Evaluating 5 configuration(s) 
## INFO  [15:44:00.836] Result of batch 4: 
## INFO  [15:44:00.840]  classif.rpart.cp classif.rpart.minsplit subsample.frac bracket bracket_stage 
## INFO  [15:44:00.840]           0.08650                      6         0.3333       1             0 
## INFO  [15:44:00.840]           0.07491                      9         0.3333       1             0 
## INFO  [15:44:00.840]           0.06716                      6         0.3333       1             0 
## INFO  [15:44:00.840]           0.06218                      9         0.3333       1             0 
## INFO  [15:44:00.840]           0.03785                      4         0.3333       1             0 
## INFO  [15:44:00.840]  budget_scaled budget_real n_configs classif.ce 
## INFO  [15:44:00.840]          3.333      0.3333         5       0.04 
## INFO  [15:44:00.840]          3.333      0.3333         5       0.04 
## INFO  [15:44:00.840]          3.333      0.3333         5       0.08 
## INFO  [15:44:00.840]          3.333      0.3333         5       0.04 
## INFO  [15:44:00.840]          3.333      0.3333         5       0.02 
## INFO  [15:44:00.840]                                 uhash 
## INFO  [15:44:00.840]  60e63f8e-572f-4be0-8e4b-8f9627ed71d2 
## INFO  [15:44:00.840]  7d8c8089-d8c8-4125-822a-f9447bacd3f6 
## INFO  [15:44:00.840]  723dafd2-4d68-4b8f-96bb-c0e7dfb29807 
## INFO  [15:44:00.840]  b1af76eb-7a12-4227-90af-127b8faed21c 
## INFO  [15:44:00.840]  ab81695e-7a85-42af-8b48-19118f4c5714 
## INFO  [15:44:00.842] Training 1 configs with budget of 1 for each 
## INFO  [15:44:00.846] Evaluating 1 configuration(s) 
## INFO  [15:44:01.165] Result of batch 5: 
## INFO  [15:44:01.168]  classif.rpart.cp classif.rpart.minsplit subsample.frac bracket bracket_stage 
## INFO  [15:44:01.168]           0.03785                      4              1       1             1 
## INFO  [15:44:01.168]  budget_scaled budget_real n_configs classif.ce 
## INFO  [15:44:01.168]             10           1         1       0.04 
## INFO  [15:44:01.168]                                 uhash 
## INFO  [15:44:01.168]  bf45e88a-5787-4d4b-a82d-470cf71fd9f5 
## INFO  [15:44:01.170] Start evaluation of bracket 3 
## INFO  [15:44:01.175] Training 3 configs with budget of 1 for each 
## INFO  [15:44:01.178] Evaluating 3 configuration(s) 
## INFO  [15:44:02.001] Result of batch 6: 
## INFO  [15:44:02.004]  classif.rpart.cp classif.rpart.minsplit subsample.frac bracket bracket_stage 
## INFO  [15:44:02.004]           0.02724                     10              1       0             0 
## INFO  [15:44:02.004]           0.05689                      3              1       0             0 
## INFO  [15:44:02.004]           0.09141                      4              1       0             0 
## INFO  [15:44:02.004]  budget_scaled budget_real n_configs classif.ce 
## INFO  [15:44:02.004]             10           1         3       0.04 
## INFO  [15:44:02.004]             10           1         3       0.04 
## INFO  [15:44:02.004]             10           1         3       0.04 
## INFO  [15:44:02.004]                                 uhash 
## INFO  [15:44:02.004]  b385920a-ceca-4584-a8d9-c5098e89054b 
## INFO  [15:44:02.004]  22821db8-5a42-4f99-9bf7-5502cc128076 
## INFO  [15:44:02.004]  d1b688fc-b4fe-461a-8a65-399b52668a78 
## INFO  [15:44:02.035] Finished optimizing after 22 evaluation(s) 
## INFO  [15:44:02.037] Result: 
## INFO  [15:44:02.040]  classif.rpart.cp classif.rpart.minsplit subsample.frac learner_param_vals 
## INFO  [15:44:02.040]            0.0849                      3         0.1111          <list[6]> 
## INFO  [15:44:02.040]   x_domain classif.ce 
## INFO  [15:44:02.040]  <list[3]>       0.02
```

```
##    classif.rpart.cp classif.rpart.minsplit subsample.frac learner_param_vals
## 1:           0.0849                      3         0.1111          <list[6]>
##     x_domain classif.ce
## 1: <list[3]>       0.02
```

To receive the results of each sampled configuration, we simply run the following snippet.


```r
instance$archive$data()[, c(
  "subsample.frac",
  "classif.rpart.cp",
  "classif.rpart.minsplit",
  "classif.ce"
), with = FALSE]
```

```
##     subsample.frac classif.rpart.cp classif.rpart.minsplit classif.ce
##  1:         0.1111          0.02533                      3       0.06
##  2:         0.1111          0.07348                      5       0.08
##  3:         0.1111          0.08490                      3       0.02
##  4:         0.1111          0.05026                      6       0.02
##  5:         0.1111          0.03940                      4       0.08
##  6:         0.1111          0.02540                      7       0.02
##  7:         0.1111          0.01200                      4       0.10
##  8:         0.1111          0.03961                      4       0.14
##  9:         0.1111          0.05762                      6       0.04
## 10:         0.3333          0.08490                      3       0.02
## 11:         0.3333          0.05026                      6       0.04
## 12:         0.3333          0.02540                      7       0.06
## 13:         1.0000          0.08490                      3       0.04
## 14:         0.3333          0.08650                      6       0.04
## 15:         0.3333          0.07491                      9       0.04
## 16:         0.3333          0.06716                      6       0.08
## 17:         0.3333          0.06218                      9       0.04
## 18:         0.3333          0.03785                      4       0.02
## 19:         1.0000          0.03785                      4       0.04
## 20:         1.0000          0.02724                     10       0.04
## 21:         1.0000          0.05689                      3       0.04
## 22:         1.0000          0.09141                      4       0.04
##     subsample.frac classif.rpart.cp classif.rpart.minsplit classif.ce
```

You can access the best found configuration through the instance object.


```r
instance$result
```

```
##    classif.rpart.cp classif.rpart.minsplit subsample.frac learner_param_vals
## 1:           0.0849                      3         0.1111          <list[6]>
##     x_domain classif.ce
## 1: <list[3]>       0.02
```

```r
instance$result_learner_param_vals
```

```
## $subsample.frac
## [1] 0.1111
## 
## $subsample.stratify
## [1] FALSE
## 
## $subsample.replace
## [1] FALSE
## 
## $classif.rpart.xval
## [1] 0
## 
## $classif.rpart.cp
## [1] 0.0849
## 
## $classif.rpart.minsplit
## [1] 3
```

```r
instance$result_y
```

```
## classif.ce 
##       0.02
```

If you are familiar with the original paper, you may have wondered how we just used Hyperband with a parameter ranging from `0.1` to `1.0` [@Li2016].
The answer is, with the help the internal rescaling of the budget parameter.
[mlr3hyperband](https://mlr3hyperband.mlr-org.com) automatically divides the budget parameters boundaries with its lower bound, ending up with a budget range starting again at `1`, like it is the case originally.
If we want an overview of what bracket layout Hyperband created and how the rescaling in each bracket worked, we can print a compact table to see this information.


```r
unique(instance$archive$data()[, .(bracket, bracket_stage, budget_scaled, budget_real, n_configs)])
```

```
##    bracket bracket_stage budget_scaled budget_real n_configs
## 1:       2             0         1.111      0.1111         9
## 2:       2             1         3.333      0.3333         3
## 3:       2             2        10.000      1.0000         1
## 4:       1             0         3.333      0.3333         5
## 5:       1             1        10.000      1.0000         1
## 6:       0             0        10.000      1.0000         3
```

In the traditional way, Hyperband uses uniform sampling to receive a configuration sample at the start of each bracket.
But it is also possible to define a custom [`Sampler`](https://paradox.mlr-org.com/reference/Sampler.html) for each hyperparameter.


```r
library(mlr3learners)
set.seed(123)

search_space = ParamSet$new(list(
  ParamInt$new("nrounds", lower = 1, upper = 16, tag = "budget"),
  ParamDbl$new("eta",     lower = 0, upper = 1),
  ParamFct$new("booster", levels = c("gbtree", "gblinear", "dart"))
))

instance = TuningInstanceSingleCrit$new(
  task = tsk("iris"),
  learner = lrn("classif.xgboost"),
  resampling = rsmp("holdout"),
  measure = msr("classif.ce"),
  terminator = trm("none"), # hyperband terminates itself
  search_space = search_space
)

# beta distribution with alpha = 2 and beta = 5
# categorical distribution with custom probabilities
sampler = SamplerJointIndep$new(list(
  Sampler1DRfun$new(search_space$params[[2]], function(n) rbeta(n, 2, 5)),
  Sampler1DCateg$new(search_space$params[[3]], prob = c(0.2, 0.3, 0.5))
))
```

Then, the defined sampler has to be given as an argument during instance creation.
Afterwards, the usual tuning can proceed.


```r
tuner = tnr("hyperband", eta = 2, sampler = sampler)
tuner$optimize(instance)
```

```
## INFO  [15:44:02.582] Starting to optimize 3 parameter(s) with '<TunerHyperband>' and '<TerminatorNone>' 
## INFO  [15:44:02.585] Amount of brackets to be evaluated = 5,  
## INFO  [15:44:02.587] Start evaluation of bracket 1 
## INFO  [15:44:02.593] Training 16 configs with budget of 1 for each 
## INFO  [15:44:02.597] Evaluating 16 configuration(s) 
## INFO  [15:44:06.402] Result of batch 1: 
## INFO  [15:44:06.406]      eta  booster nrounds bracket bracket_stage budget_scaled budget_real 
## INFO  [15:44:06.406]  0.16633 gblinear       1       4             0             1           1 
## INFO  [15:44:06.406]  0.53672 gblinear       1       4             0             1           1 
## INFO  [15:44:06.406]  0.23163     dart       1       4             0             1           1 
## INFO  [15:44:06.406]  0.09921     dart       1       4             0             1           1 
## INFO  [15:44:06.406]  0.32375     dart       1       4             0             1           1 
## INFO  [15:44:06.406]  0.25848 gblinear       1       4             0             1           1 
## INFO  [15:44:06.406]  0.28688 gblinear       1       4             0             1           1 
## INFO  [15:44:06.406]  0.36995   gbtree       1       4             0             1           1 
## INFO  [15:44:06.406]  0.21663 gblinear       1       4             0             1           1 
## INFO  [15:44:06.406]  0.43376     dart       1       4             0             1           1 
## INFO  [15:44:06.406]  0.24324 gblinear       1       4             0             1           1 
## INFO  [15:44:06.406]  0.35749     dart       1       4             0             1           1 
## INFO  [15:44:06.406]  0.38180     dart       1       4             0             1           1 
## INFO  [15:44:06.406]  0.22436     dart       1       4             0             1           1 
## INFO  [15:44:06.406]  0.57168     dart       1       4             0             1           1 
## INFO  [15:44:06.406]  0.52773   gbtree       1       4             0             1           1 
## INFO  [15:44:06.406]  n_configs classif.ce                                uhash 
## INFO  [15:44:06.406]         16       0.74 cf9ffa48-39ab-4352-8bd4-c97f7780a4fd 
## INFO  [15:44:06.406]         16       0.42 575b791b-24f1-4498-86a3-f28d4fcb9d66 
## INFO  [15:44:06.406]         16       0.04 5edb2666-ba7c-4567-91f9-a48302680558 
## INFO  [15:44:06.406]         16       0.04 5caafa67-8cfc-46d0-9c08-04752c9d6e9f 
## INFO  [15:44:06.406]         16       0.04 fdbe2bd6-3d77-4764-baf5-de37510ad2d9 
## INFO  [15:44:06.406]         16       0.74 c8797865-7111-4378-a6ce-c03febd5310f 
## INFO  [15:44:06.406]         16       0.74 c5799909-d109-4251-bb75-2bd4c42af2b9 
## INFO  [15:44:06.406]         16       0.04 f674523f-3226-48f2-81da-76a07220930e 
## INFO  [15:44:06.406]         16       0.74 55dfdf11-6893-4886-b90b-6e41fe2a0a96 
## INFO  [15:44:06.406]         16       0.04 e6e4936a-c10d-484f-9af9-1a2866a388e8 
## INFO  [15:44:06.406]         16       0.74 73cbb934-52bf-4c67-addd-1bad1f00a2e1 
## INFO  [15:44:06.406]         16       0.04 db2ec9df-956a-4553-8458-0491da565f5a 
## INFO  [15:44:06.406]         16       0.04 711677f6-67fc-49bf-bf47-83e2d70fd267 
## INFO  [15:44:06.406]         16       0.04 40c22684-5d2b-4e5c-ae0f-7f9c13cd6c42 
## INFO  [15:44:06.406]         16       0.04 42bbdb1d-2411-4218-9fd6-f35d91954255 
## INFO  [15:44:06.406]         16       0.04 6518a753-a4cf-4a96-b041-fdcdffee1149 
## INFO  [15:44:06.408] Training 8 configs with budget of 2 for each 
## INFO  [15:44:06.413] Evaluating 8 configuration(s) 
## INFO  [15:44:07.647] Result of batch 2: 
## INFO  [15:44:07.650]      eta booster nrounds bracket bracket_stage budget_scaled budget_real 
## INFO  [15:44:07.650]  0.23163    dart       2       4             1             2           2 
## INFO  [15:44:07.650]  0.09921    dart       2       4             1             2           2 
## INFO  [15:44:07.650]  0.32375    dart       2       4             1             2           2 
## INFO  [15:44:07.650]  0.36995  gbtree       2       4             1             2           2 
## INFO  [15:44:07.650]  0.43376    dart       2       4             1             2           2 
## INFO  [15:44:07.650]  0.35749    dart       2       4             1             2           2 
## INFO  [15:44:07.650]  0.38180    dart       2       4             1             2           2 
## INFO  [15:44:07.650]  0.22436    dart       2       4             1             2           2 
## INFO  [15:44:07.650]  n_configs classif.ce                                uhash 
## INFO  [15:44:07.650]          8       0.04 b3a6d281-3737-4564-988d-afa9649d5ef0 
## INFO  [15:44:07.650]          8       0.04 02db7266-1b3f-49f1-bd93-934b6a0b8b3e 
## INFO  [15:44:07.650]          8       0.04 8e736cb7-08b9-4aee-9950-c29fb1bae855 
## INFO  [15:44:07.650]          8       0.04 d3c02720-94d8-4f0f-940e-9a1519a092da 
## INFO  [15:44:07.650]          8       0.04 2ff1e727-93a3-4146-8782-fe462f9ea811 
## INFO  [15:44:07.650]          8       0.04 bdd91f8b-9fc7-4fae-b087-e6af08a16523 
## INFO  [15:44:07.650]          8       0.04 de5172af-d7e2-4044-b530-7e65afe36531 
## INFO  [15:44:07.650]          8       0.04 3eccf8ae-91f1-435b-8097-3432e0349cc0 
## INFO  [15:44:07.652] Training 4 configs with budget of 4 for each 
## INFO  [15:44:07.656] Evaluating 4 configuration(s) 
## INFO  [15:44:08.334] Result of batch 3: 
## INFO  [15:44:08.338]      eta booster nrounds bracket bracket_stage budget_scaled budget_real 
## INFO  [15:44:08.338]  0.23163    dart       4       4             2             4           4 
## INFO  [15:44:08.338]  0.09921    dart       4       4             2             4           4 
## INFO  [15:44:08.338]  0.32375    dart       4       4             2             4           4 
## INFO  [15:44:08.338]  0.36995  gbtree       4       4             2             4           4 
## INFO  [15:44:08.338]  n_configs classif.ce                                uhash 
## INFO  [15:44:08.338]          4       0.04 db107b4b-5872-4168-bed5-13f82b9f43a4 
## INFO  [15:44:08.338]          4       0.04 d096821a-d986-4008-bd88-3ffb152428ae 
## INFO  [15:44:08.338]          4       0.04 26455c10-1da9-412e-911b-4b87758d0a25 
## INFO  [15:44:08.338]          4       0.04 ee00cd09-c3bf-42d5-948e-11c99cab0d8e 
## INFO  [15:44:08.340] Training 2 configs with budget of 8 for each 
## INFO  [15:44:08.344] Evaluating 2 configuration(s) 
## INFO  [15:44:08.733] Result of batch 4: 
## INFO  [15:44:08.736]      eta booster nrounds bracket bracket_stage budget_scaled budget_real 
## INFO  [15:44:08.736]  0.23163    dart       8       4             3             8           8 
## INFO  [15:44:08.736]  0.09921    dart       8       4             3             8           8 
## INFO  [15:44:08.736]  n_configs classif.ce                                uhash 
## INFO  [15:44:08.736]          2       0.04 2a19f483-c790-4f0a-bb6e-b59f379a648f 
## INFO  [15:44:08.736]          2       0.04 3e97c442-2d6e-4e5b-9be6-a6daccd8c1c3 
## INFO  [15:44:08.738] Training 1 configs with budget of 16 for each 
## INFO  [15:44:08.743] Evaluating 1 configuration(s) 
## INFO  [15:44:08.993] Result of batch 5: 
## INFO  [15:44:08.996]     eta booster nrounds bracket bracket_stage budget_scaled budget_real 
## INFO  [15:44:08.996]  0.2316    dart      16       4             4            16          16 
## INFO  [15:44:08.996]  n_configs classif.ce                                uhash 
## INFO  [15:44:08.996]          1       0.04 f73fdaa4-46ce-4af8-8b29-959939b84cb9 
## INFO  [15:44:09.002] Start evaluation of bracket 2 
## INFO  [15:44:09.008] Training 10 configs with budget of 2 for each 
## INFO  [15:44:09.012] Evaluating 10 configuration(s) 
## INFO  [15:44:10.566] Result of batch 6: 
## INFO  [15:44:10.570]      eta  booster nrounds bracket bracket_stage budget_scaled budget_real 
## INFO  [15:44:10.570]  0.17165 gblinear       2       3             0             2           2 
## INFO  [15:44:10.570]  0.33565   gbtree       2       3             0             2           2 
## INFO  [15:44:10.570]  0.30172   gbtree       2       3             0             2           2 
## INFO  [15:44:10.570]  0.12918     dart       2       3             0             2           2 
## INFO  [15:44:10.570]  0.27153     dart       2       3             0             2           2 
## INFO  [15:44:10.570]  0.38573 gblinear       2       3             0             2           2 
## INFO  [15:44:10.570]  0.29412 gblinear       2       3             0             2           2 
## INFO  [15:44:10.570]  0.20787     dart       2       3             0             2           2 
## INFO  [15:44:10.570]  0.03459 gblinear       2       3             0             2           2 
## INFO  [15:44:10.570]  0.56669 gblinear       2       3             0             2           2 
## INFO  [15:44:10.570]  n_configs classif.ce                                uhash 
## INFO  [15:44:10.570]         10       0.70 d40752ac-411e-43f3-815a-538db1e90d9c 
## INFO  [15:44:10.570]         10       0.04 1e10b3d9-b632-433e-a602-3f5bad6b7f4c 
## INFO  [15:44:10.570]         10       0.04 735c60e6-58c8-4c3e-a5f0-f4d9f13935bb 
## INFO  [15:44:10.570]         10       0.04 3b5d15ef-e2aa-4dbf-a5c1-3d357cad2857 
## INFO  [15:44:10.570]         10       0.04 b98676be-3428-4768-b5fe-87a27a67be32 
## INFO  [15:44:10.570]         10       0.42 57c5eb2f-be0c-4e73-93e1-1b12daa42311 
## INFO  [15:44:10.570]         10       0.44 f12e2b75-4d18-4da9-8b5b-da1262526ee2 
## INFO  [15:44:10.570]         10       0.04 210d5559-c3e4-4ae7-a72e-3fd031eca60b 
## INFO  [15:44:10.570]         10       0.74 8d2457f3-8abd-4d26-a0a2-537e266fe974 
## INFO  [15:44:10.570]         10       0.42 db833dbd-3f97-4ed6-8135-aa7eba03db6e 
## INFO  [15:44:10.572] Training 5 configs with budget of 4 for each 
## INFO  [15:44:10.576] Evaluating 5 configuration(s) 
## INFO  [15:44:11.687] Result of batch 7: 
## INFO  [15:44:11.691]     eta booster nrounds bracket bracket_stage budget_scaled budget_real 
## INFO  [15:44:11.691]  0.3356  gbtree       4       3             1             4           4 
## INFO  [15:44:11.691]  0.3017  gbtree       4       3             1             4           4 
## INFO  [15:44:11.691]  0.1292    dart       4       3             1             4           4 
## INFO  [15:44:11.691]  0.2715    dart       4       3             1             4           4 
## INFO  [15:44:11.691]  0.2079    dart       4       3             1             4           4 
## INFO  [15:44:11.691]  n_configs classif.ce                                uhash 
## INFO  [15:44:11.691]          5       0.04 1976ea93-f620-4ccb-bc17-01d860395976 
## INFO  [15:44:11.691]          5       0.04 31c3fde4-1fec-4bab-8b6a-c8efc98c5f95 
## INFO  [15:44:11.691]          5       0.04 f620406f-4b39-4d62-956c-041714413e6a 
## INFO  [15:44:11.691]          5       0.04 83a541ec-b284-4f55-8264-df3ce0e25b93 
## INFO  [15:44:11.691]          5       0.04 9c4790cc-261a-4a98-bde6-c1fc64e31ac1 
## INFO  [15:44:11.693] Training 2 configs with budget of 8 for each 
## INFO  [15:44:11.697] Evaluating 2 configuration(s) 
## INFO  [15:44:12.073] Result of batch 8: 
## INFO  [15:44:12.077]     eta booster nrounds bracket bracket_stage budget_scaled budget_real 
## INFO  [15:44:12.077]  0.3356  gbtree       8       3             2             8           8 
## INFO  [15:44:12.077]  0.3017  gbtree       8       3             2             8           8 
## INFO  [15:44:12.077]  n_configs classif.ce                                uhash 
## INFO  [15:44:12.077]          2       0.04 e3c06b5c-8890-4e38-a8d0-1611f445f5fe 
## INFO  [15:44:12.077]          2       0.04 af402b90-aee8-4bb5-9bcc-c216f27e61c8 
## INFO  [15:44:12.078] Training 1 configs with budget of 16 for each 
## INFO  [15:44:12.082] Evaluating 1 configuration(s) 
## INFO  [15:44:12.331] Result of batch 9: 
## INFO  [15:44:12.335]     eta booster nrounds bracket bracket_stage budget_scaled budget_real 
## INFO  [15:44:12.335]  0.3356  gbtree      16       3             3            16          16 
## INFO  [15:44:12.335]  n_configs classif.ce                                uhash 
## INFO  [15:44:12.335]          1       0.04 f79310f9-76fa-485f-852f-5284e5474ffe 
## INFO  [15:44:12.337] Start evaluation of bracket 3 
## INFO  [15:44:12.342] Training 7 configs with budget of 4 for each 
## INFO  [15:44:12.345] Evaluating 7 configuration(s) 
## INFO  [15:44:13.444] Result of batch 10: 
## INFO  [15:44:13.448]      eta  booster nrounds bracket bracket_stage budget_scaled budget_real 
## INFO  [15:44:13.448]  0.41312 gblinear       4       2             0             4           4 
## INFO  [15:44:13.448]  0.21633     dart       4       2             0             4           4 
## INFO  [15:44:13.448]  0.52311     dart       4       2             0             4           4 
## INFO  [15:44:13.448]  0.21596     dart       4       2             0             4           4 
## INFO  [15:44:13.448]  0.54437   gbtree       4       2             0             4           4 
## INFO  [15:44:13.448]  0.11852     dart       4       2             0             4           4 
## INFO  [15:44:13.448]  0.09508     dart       4       2             0             4           4 
## INFO  [15:44:13.448]  n_configs classif.ce                                uhash 
## INFO  [15:44:13.448]          7       0.42 ae83e621-ce19-4fd5-aba2-5842379c44d7 
## INFO  [15:44:13.448]          7       0.04 8c3af84c-1851-4544-b255-d1ad6ff934a9 
## INFO  [15:44:13.448]          7       0.04 d433ff35-0ec9-45ef-85c8-aa0879f4d5fa 
## INFO  [15:44:13.448]          7       0.04 427de415-90d7-4ac7-b6c2-bd0b9375e733 
## INFO  [15:44:13.448]          7       0.04 093be16d-59ee-47ee-a2f2-119322bf60d9 
## INFO  [15:44:13.448]          7       0.04 f04e44fe-84b0-4df7-a050-56f9781f2976 
## INFO  [15:44:13.448]          7       0.04 121564de-9043-41ef-9cf9-30e79755f57d 
## INFO  [15:44:13.450] Training 3 configs with budget of 8 for each 
## INFO  [15:44:13.455] Evaluating 3 configuration(s) 
## INFO  [15:44:13.977] Result of batch 11: 
## INFO  [15:44:13.980]     eta booster nrounds bracket bracket_stage budget_scaled budget_real 
## INFO  [15:44:13.980]  0.2163    dart       8       2             1             8           8 
## INFO  [15:44:13.980]  0.5231    dart       8       2             1             8           8 
## INFO  [15:44:13.980]  0.2160    dart       8       2             1             8           8 
## INFO  [15:44:13.980]  n_configs classif.ce                                uhash 
## INFO  [15:44:13.980]          3       0.04 299f75b6-caa1-4cd1-966c-d8b9f6c282dd 
## INFO  [15:44:13.980]          3       0.04 a65225d0-5ffd-49a6-97bc-8fb0ad921355 
## INFO  [15:44:13.980]          3       0.04 54f64cff-f6f8-45cf-b44e-150b4d837e8c 
## INFO  [15:44:13.982] Training 1 configs with budget of 16 for each 
## INFO  [15:44:13.987] Evaluating 1 configuration(s) 
## INFO  [15:44:14.230] Result of batch 12: 
## INFO  [15:44:14.234]     eta booster nrounds bracket bracket_stage budget_scaled budget_real 
## INFO  [15:44:14.234]  0.2163    dart      16       2             2            16          16 
## INFO  [15:44:14.234]  n_configs classif.ce                                uhash 
## INFO  [15:44:14.234]          1       0.04 907745cf-37cf-49b6-9a79-0c9ba7ce937e 
## INFO  [15:44:14.236] Start evaluation of bracket 4 
## INFO  [15:44:14.240] Training 5 configs with budget of 8 for each 
## INFO  [15:44:14.244] Evaluating 5 configuration(s) 
## INFO  [15:44:15.076] Result of batch 13: 
## INFO  [15:44:15.081]     eta  booster nrounds bracket bracket_stage budget_scaled budget_real 
## INFO  [15:44:15.081]  0.2462   gbtree       8       1             0             8           8 
## INFO  [15:44:15.081]  0.5226 gblinear       8       1             0             8           8 
## INFO  [15:44:15.081]  0.1413 gblinear       8       1             0             8           8 
## INFO  [15:44:15.081]  0.1950     dart       8       1             0             8           8 
## INFO  [15:44:15.081]  0.4708 gblinear       8       1             0             8           8 
## INFO  [15:44:15.081]  n_configs classif.ce                                uhash 
## INFO  [15:44:15.081]          5       0.04 86c3ba76-65d8-42aa-8138-372e86697c3a 
## INFO  [15:44:15.081]          5       0.42 1bdd58d5-27d6-47d9-ab08-b94152ddbe5b 
## INFO  [15:44:15.081]          5       0.42 221854f8-2f92-43e4-86df-c6963cb197de 
## INFO  [15:44:15.081]          5       0.04 b2bb735c-1edb-40d0-a667-334bd5de67ba 
## INFO  [15:44:15.081]          5       0.42 9c46dcfb-9416-47bd-85db-b6c4ba7b22f3 
## INFO  [15:44:15.083] Training 2 configs with budget of 16 for each 
## INFO  [15:44:15.087] Evaluating 2 configuration(s) 
## INFO  [15:44:15.471] Result of batch 14: 
## INFO  [15:44:15.475]     eta booster nrounds bracket bracket_stage budget_scaled budget_real 
## INFO  [15:44:15.475]  0.2462  gbtree      16       1             1            16          16 
## INFO  [15:44:15.475]  0.1950    dart      16       1             1            16          16 
## INFO  [15:44:15.475]  n_configs classif.ce                                uhash 
## INFO  [15:44:15.475]          2       0.04 3d41a090-975d-4b58-9f83-15d8ce591e25 
## INFO  [15:44:15.475]          2       0.04 26461fb8-463d-41b4-b728-784c1087aba3 
## INFO  [15:44:15.476] Start evaluation of bracket 5 
## INFO  [15:44:15.481] Training 5 configs with budget of 16 for each 
## INFO  [15:44:15.485] Evaluating 5 configuration(s) 
## INFO  [15:44:16.318] Result of batch 15: 
## INFO  [15:44:16.322]      eta booster nrounds bracket bracket_stage budget_scaled budget_real 
## INFO  [15:44:16.322]  0.08993    dart      16       0             0            16          16 
## INFO  [15:44:16.322]  0.42262    dart      16       0             0            16          16 
## INFO  [15:44:16.322]  0.09600  gbtree      16       0             0            16          16 
## INFO  [15:44:16.322]  0.17779    dart      16       0             0            16          16 
## INFO  [15:44:16.322]  0.61866    dart      16       0             0            16          16 
## INFO  [15:44:16.322]  n_configs classif.ce                                uhash 
## INFO  [15:44:16.322]          5       0.04 7eba7930-1aa6-4c75-a5a5-7b2e677a441c 
## INFO  [15:44:16.322]          5       0.04 0b15e871-8d7d-4f27-98cf-38d6b28fc54c 
## INFO  [15:44:16.322]          5       0.04 45011580-319a-48d3-af92-e84196c5be86 
## INFO  [15:44:16.322]          5       0.04 872488e4-f98e-42a0-a7fe-52cb4ce89cce 
## INFO  [15:44:16.322]          5       0.04 0fb731b0-7e63-4eb0-91a8-1e01a51ce654 
## INFO  [15:44:16.330] Finished optimizing after 72 evaluation(s) 
## INFO  [15:44:16.332] Result: 
## INFO  [15:44:16.334]  nrounds    eta booster learner_param_vals  x_domain classif.ce 
## INFO  [15:44:16.334]        1 0.2316    dart          <list[4]> <list[3]>       0.04
```

```
##    nrounds    eta booster learner_param_vals  x_domain classif.ce
## 1:       1 0.2316    dart          <list[4]> <list[3]>       0.04
```

```r
instance$result
```

```
##    nrounds    eta booster learner_param_vals  x_domain classif.ce
## 1:       1 0.2316    dart          <list[4]> <list[3]>       0.04
```

Furthermore, we extended the original algorithm, to make it also possible to use [mlr3hyperband](https://mlr3hyperband.mlr-org.com) for multi-objective optimization.
To do this, simply specify more measures in the [`TuningInstanceMultiCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceMultiCrit.html) and run the rest as usual.


```r
instance = TuningInstanceMultiCrit$new(
  task = tsk("pima"),
  learner = lrn("classif.xgboost"),
  resampling = rsmp("holdout"),
  measures = msrs(c("classif.tpr", "classif.fpr")),
  terminator = trm("none"), # hyperband terminates itself
  search_space = search_space
)

tuner = tnr("hyperband", eta = 4)
tuner$optimize(instance)
```

```
## INFO  [15:44:16.775] Starting to optimize 3 parameter(s) with '<TunerHyperband>' and '<TerminatorNone>' 
## INFO  [15:44:16.794] Amount of brackets to be evaluated = 3,  
## INFO  [15:44:16.796] Start evaluation of bracket 1 
## INFO  [15:44:16.801] Training 16 configs with budget of 1 for each 
## INFO  [15:44:16.805] Evaluating 16 configuration(s) 
## INFO  [15:44:19.423] Result of batch 1: 
## INFO  [15:44:19.428]      eta  booster nrounds bracket bracket_stage budget_scaled budget_real 
## INFO  [15:44:19.428]  0.20737 gblinear       1       2             0             1           1 
## INFO  [15:44:19.428]  0.45924   gbtree       1       2             0             1           1 
## INFO  [15:44:19.428]  0.24150 gblinear       1       2             0             1           1 
## INFO  [15:44:19.428]  0.11869   gbtree       1       2             0             1           1 
## INFO  [15:44:19.428]  0.07247   gbtree       1       2             0             1           1 
## INFO  [15:44:19.428]  0.69099     dart       1       2             0             1           1 
## INFO  [15:44:19.428]  0.28696     dart       1       2             0             1           1 
## INFO  [15:44:19.428]  0.14941     dart       1       2             0             1           1 
## INFO  [15:44:19.428]  0.97243   gbtree       1       2             0             1           1 
## INFO  [15:44:19.428]  0.41051 gblinear       1       2             0             1           1 
## INFO  [15:44:19.428]  0.40181     dart       1       2             0             1           1 
## INFO  [15:44:19.428]  0.64856     dart       1       2             0             1           1 
## INFO  [15:44:19.428]  0.91631 gblinear       1       2             0             1           1 
## INFO  [15:44:19.428]  0.21666   gbtree       1       2             0             1           1 
## INFO  [15:44:19.428]  0.54800 gblinear       1       2             0             1           1 
## INFO  [15:44:19.428]  0.72005 gblinear       1       2             0             1           1 
## INFO  [15:44:19.428]  n_configs classif.tpr classif.fpr                                uhash 
## INFO  [15:44:19.428]         16      0.0000      0.0000 3c82fc40-a18a-4628-b8dc-985af17eb5b5 
## INFO  [15:44:19.428]         16      0.7531      0.2571 38d01c0b-8f2c-4663-bd01-1c364b6d2173 
## INFO  [15:44:19.428]         16      0.0000      0.0000 fdd91d05-8999-4fc9-8d88-d3dd0a44cb2d 
## INFO  [15:44:19.428]         16      0.7407      0.2457 f7d4da96-cfab-4811-bcf2-012674589d13 
## INFO  [15:44:19.428]         16      0.7407      0.2571 69bd80bf-91a0-473b-aec9-8674259d28b7 
## INFO  [15:44:19.428]         16      0.7407      0.2629 8d272c6f-c320-444d-95b5-2206d9de5b1f 
## INFO  [15:44:19.428]         16      0.7531      0.2629 aab11910-086d-4d22-99c1-3649c694aef9 
## INFO  [15:44:19.428]         16      0.7407      0.2514 dd04223b-c034-4841-9f4f-6936ae89f701 
## INFO  [15:44:19.428]         16      0.7531      0.2571 0b2057c6-809b-423c-9058-a06ad052809d 
## INFO  [15:44:19.428]         16      0.0000      0.0000 eddb25c3-274b-4898-98fa-aab8d4c8c786 
## INFO  [15:44:19.428]         16      0.7407      0.2457 a0cd3dc8-3f0f-4b6e-8417-f3922112ebed 
## INFO  [15:44:19.428]         16      0.7531      0.2686 74744094-3d05-4aeb-bb7e-2d11a184837a 
## INFO  [15:44:19.428]         16      0.0000      0.0000 ccd4654b-73fd-4534-93fe-95765db14077 
## INFO  [15:44:19.428]         16      0.7407      0.2571 4a77d5da-9bd3-4004-b1e9-0ccf1d7c4466 
## INFO  [15:44:19.428]         16      0.0000      0.0000 edaa6048-a2fd-4065-937d-e3e37520be5f 
## INFO  [15:44:19.428]         16      0.0000      0.0000 1ec75795-ca98-46aa-a686-b9455bbe1a3d 
## INFO  [15:44:19.429] Training 4 configs with budget of 4 for each 
## INFO  [15:44:19.436] Evaluating 4 configuration(s) 
## INFO  [15:44:20.180] Result of batch 2: 
## INFO  [15:44:20.184]     eta  booster nrounds bracket bracket_stage budget_scaled budget_real 
## INFO  [15:44:20.184]  0.4592   gbtree       4       2             1             4           4 
## INFO  [15:44:20.184]  0.1187   gbtree       4       2             1             4           4 
## INFO  [15:44:20.184]  0.5480 gblinear       4       2             1             4           4 
## INFO  [15:44:20.184]  0.7201 gblinear       4       2             1             4           4 
## INFO  [15:44:20.184]  n_configs classif.tpr classif.fpr                                uhash 
## INFO  [15:44:20.184]          4     0.66667     0.18286 614d9eea-b5fb-4b0a-8baa-64106870ccbd 
## INFO  [15:44:20.184]          4     0.72840     0.22286 7cb143e1-826c-4da5-a988-0219f3610749 
## INFO  [15:44:20.184]          4     0.06173     0.02857 af3fd521-d99b-468d-a5f8-9c26808de8d4 
## INFO  [15:44:20.184]          4     0.09877     0.04571 c8703ae4-18cb-41fc-befe-708000e1386a 
## INFO  [15:44:20.186] Training 1 configs with budget of 16 for each 
## INFO  [15:44:20.191] Evaluating 1 configuration(s) 
## INFO  [15:44:20.457] Result of batch 3: 
## INFO  [15:44:20.461]     eta booster nrounds bracket bracket_stage budget_scaled budget_real 
## INFO  [15:44:20.461]  0.1187  gbtree      16       2             2            16          16 
## INFO  [15:44:20.461]  n_configs classif.tpr classif.fpr                                uhash 
## INFO  [15:44:20.461]          1      0.5926      0.1543 5bcbfb63-e028-48ac-907a-ed26c249ba7e 
## INFO  [15:44:20.463] Start evaluation of bracket 2 
## INFO  [15:44:20.467] Training 6 configs with budget of 4 for each 
## INFO  [15:44:20.471] Evaluating 6 configuration(s) 
## INFO  [15:44:21.670] Result of batch 4: 
## INFO  [15:44:21.674]      eta  booster nrounds bracket bracket_stage budget_scaled budget_real 
## INFO  [15:44:21.674]  0.98871     dart       4       1             0             4           4 
## INFO  [15:44:21.674]  0.06475   gbtree       4       1             0             4           4 
## INFO  [15:44:21.674]  0.15766 gblinear       4       1             0             4           4 
## INFO  [15:44:21.674]  0.78535   gbtree       4       1             0             4           4 
## INFO  [15:44:21.674]  0.54219     dart       4       1             0             4           4 
## INFO  [15:44:21.674]  0.41655 gblinear       4       1             0             4           4 
## INFO  [15:44:21.674]  n_configs classif.tpr classif.fpr                                uhash 
## INFO  [15:44:21.674]          6     0.61728     0.17714 070668f8-022d-46ca-9356-35a2651ee0b5 
## INFO  [15:44:21.674]          6     0.64198     0.18286 28576fda-b674-4856-b45d-9b527f7f39d7 
## INFO  [15:44:21.674]          6     0.00000     0.00000 bfa8833b-30e0-4f51-9bd5-72a612f529ba 
## INFO  [15:44:21.674]          6     0.66667     0.18857 3e9f131f-f7a1-4e15-8a00-6a8493c737de 
## INFO  [15:44:21.674]          6     0.60494     0.15429 ba208d14-eda3-474a-b838-3b97551a59ac 
## INFO  [15:44:21.674]          6     0.03704     0.01143 1402dad2-30c9-407e-ad10-6fb57d0475e2 
## INFO  [15:44:21.676] Training 1 configs with budget of 16 for each 
## INFO  [15:44:21.681] Evaluating 1 configuration(s) 
## INFO  [15:44:21.979] Result of batch 5: 
## INFO  [15:44:21.983]     eta booster nrounds bracket bracket_stage budget_scaled budget_real 
## INFO  [15:44:21.983]  0.7853  gbtree      16       1             1            16          16 
## INFO  [15:44:21.983]  n_configs classif.tpr classif.fpr                                uhash 
## INFO  [15:44:21.983]          1      0.6543      0.2114 4a1aedd6-99ae-4b70-a8fd-fbb6ed45419d 
## INFO  [15:44:21.985] Start evaluation of bracket 3 
## INFO  [15:44:21.990] Training 3 configs with budget of 16 for each 
## INFO  [15:44:21.993] Evaluating 3 configuration(s) 
## INFO  [15:44:22.615] Result of batch 6: 
## INFO  [15:44:22.619]     eta  booster nrounds bracket bracket_stage budget_scaled budget_real 
## INFO  [15:44:22.619]  0.5221     dart      16       0             0            16          16 
## INFO  [15:44:22.619]  0.1117   gbtree      16       0             0            16          16 
## INFO  [15:44:22.619]  0.8860 gblinear      16       0             0            16          16 
## INFO  [15:44:22.619]  n_configs classif.tpr classif.fpr                                uhash 
## INFO  [15:44:22.619]          3      0.6420      0.2171 e6f86ade-8c21-4857-9685-11005f0f1691 
## INFO  [15:44:22.619]          3      0.6543      0.1714 7ae5c493-13d2-4854-a739-c2348179b71c 
## INFO  [15:44:22.619]          3      0.4815      0.1886 1d38d2f3-5157-405b-930a-61ad6ea45f2c 
## INFO  [15:44:22.629] Finished optimizing after 31 evaluation(s) 
## INFO  [15:44:22.631] Result: 
## INFO  [15:44:22.635]  nrounds    eta  booster learner_param_vals  x_domain classif.tpr classif.fpr 
## INFO  [15:44:22.635]        1 0.2074 gblinear          <list[4]> <list[3]>     0.00000     0.00000 
## INFO  [15:44:22.635]        1 0.4592   gbtree          <list[4]> <list[3]>     0.75309     0.25714 
## INFO  [15:44:22.635]        1 0.2415 gblinear          <list[4]> <list[3]>     0.00000     0.00000 
## INFO  [15:44:22.635]        1 0.1187   gbtree          <list[4]> <list[3]>     0.74074     0.24571 
## INFO  [15:44:22.635]        1 0.9724   gbtree          <list[4]> <list[3]>     0.75309     0.25714 
## INFO  [15:44:22.635]        1 0.4105 gblinear          <list[4]> <list[3]>     0.00000     0.00000 
## INFO  [15:44:22.635]        1 0.4018     dart          <list[4]> <list[3]>     0.74074     0.24571 
## INFO  [15:44:22.635]        1 0.9163 gblinear          <list[4]> <list[3]>     0.00000     0.00000 
## INFO  [15:44:22.635]        1 0.5480 gblinear          <list[4]> <list[3]>     0.00000     0.00000 
## INFO  [15:44:22.635]        1 0.7201 gblinear          <list[4]> <list[3]>     0.00000     0.00000 
## INFO  [15:44:22.635]        4 0.4592   gbtree          <list[4]> <list[3]>     0.66667     0.18286 
## INFO  [15:44:22.635]        4 0.1187   gbtree          <list[4]> <list[3]>     0.72840     0.22286 
## INFO  [15:44:22.635]        4 0.5480 gblinear          <list[4]> <list[3]>     0.06173     0.02857 
## INFO  [15:44:22.635]        4 0.7201 gblinear          <list[4]> <list[3]>     0.09877     0.04571 
## INFO  [15:44:22.635]        4 0.1577 gblinear          <list[4]> <list[3]>     0.00000     0.00000 
## INFO  [15:44:22.635]        4 0.5422     dart          <list[4]> <list[3]>     0.60494     0.15429 
## INFO  [15:44:22.635]        4 0.4165 gblinear          <list[4]> <list[3]>     0.03704     0.01143 
## INFO  [15:44:22.635]       16 0.1117   gbtree          <list[4]> <list[3]>     0.65432     0.17143
```

```
##     nrounds    eta  booster learner_param_vals  x_domain classif.tpr
##  1:       1 0.2074 gblinear          <list[4]> <list[3]>     0.00000
##  2:       1 0.4592   gbtree          <list[4]> <list[3]>     0.75309
##  3:       1 0.2415 gblinear          <list[4]> <list[3]>     0.00000
##  4:       1 0.1187   gbtree          <list[4]> <list[3]>     0.74074
##  5:       1 0.9724   gbtree          <list[4]> <list[3]>     0.75309
##  6:       1 0.4105 gblinear          <list[4]> <list[3]>     0.00000
##  7:       1 0.4018     dart          <list[4]> <list[3]>     0.74074
##  8:       1 0.9163 gblinear          <list[4]> <list[3]>     0.00000
##  9:       1 0.5480 gblinear          <list[4]> <list[3]>     0.00000
## 10:       1 0.7201 gblinear          <list[4]> <list[3]>     0.00000
## 11:       4 0.4592   gbtree          <list[4]> <list[3]>     0.66667
## 12:       4 0.1187   gbtree          <list[4]> <list[3]>     0.72840
## 13:       4 0.5480 gblinear          <list[4]> <list[3]>     0.06173
## 14:       4 0.7201 gblinear          <list[4]> <list[3]>     0.09877
## 15:       4 0.1577 gblinear          <list[4]> <list[3]>     0.00000
## 16:       4 0.5422     dart          <list[4]> <list[3]>     0.60494
## 17:       4 0.4165 gblinear          <list[4]> <list[3]>     0.03704
## 18:      16 0.1117   gbtree          <list[4]> <list[3]>     0.65432
##     classif.fpr
##  1:     0.00000
##  2:     0.25714
##  3:     0.00000
##  4:     0.24571
##  5:     0.25714
##  6:     0.00000
##  7:     0.24571
##  8:     0.00000
##  9:     0.00000
## 10:     0.00000
## 11:     0.18286
## 12:     0.22286
## 13:     0.02857
## 14:     0.04571
## 15:     0.00000
## 16:     0.15429
## 17:     0.01143
## 18:     0.17143
```

Now the result is not a single best configuration but an estimated Pareto front.
All red points are not dominated by another parameter configuration regarding their *fpr* and *tpr* performance measures.


```r
instance$result
```

```
##     nrounds    eta  booster learner_param_vals  x_domain classif.tpr
##  1:       1 0.2074 gblinear          <list[4]> <list[3]>     0.00000
##  2:       1 0.4592   gbtree          <list[4]> <list[3]>     0.75309
##  3:       1 0.2415 gblinear          <list[4]> <list[3]>     0.00000
##  4:       1 0.1187   gbtree          <list[4]> <list[3]>     0.74074
##  5:       1 0.9724   gbtree          <list[4]> <list[3]>     0.75309
##  6:       1 0.4105 gblinear          <list[4]> <list[3]>     0.00000
##  7:       1 0.4018     dart          <list[4]> <list[3]>     0.74074
##  8:       1 0.9163 gblinear          <list[4]> <list[3]>     0.00000
##  9:       1 0.5480 gblinear          <list[4]> <list[3]>     0.00000
## 10:       1 0.7201 gblinear          <list[4]> <list[3]>     0.00000
## 11:       4 0.4592   gbtree          <list[4]> <list[3]>     0.66667
## 12:       4 0.1187   gbtree          <list[4]> <list[3]>     0.72840
## 13:       4 0.5480 gblinear          <list[4]> <list[3]>     0.06173
## 14:       4 0.7201 gblinear          <list[4]> <list[3]>     0.09877
## 15:       4 0.1577 gblinear          <list[4]> <list[3]>     0.00000
## 16:       4 0.5422     dart          <list[4]> <list[3]>     0.60494
## 17:       4 0.4165 gblinear          <list[4]> <list[3]>     0.03704
## 18:      16 0.1117   gbtree          <list[4]> <list[3]>     0.65432
##     classif.fpr
##  1:     0.00000
##  2:     0.25714
##  3:     0.00000
##  4:     0.24571
##  5:     0.25714
##  6:     0.00000
##  7:     0.24571
##  8:     0.00000
##  9:     0.00000
## 10:     0.00000
## 11:     0.18286
## 12:     0.22286
## 13:     0.02857
## 14:     0.04571
## 15:     0.00000
## 16:     0.15429
## 17:     0.01143
## 18:     0.17143
```

```r
plot(classif.tpr~classif.fpr, instance$archive$data())
points(classif.tpr~classif.fpr, instance$result, col = "red")
```

<img src="03-optimization-hyperband_files/figure-html/03-optimization-hyperband-013-1.svg" width="672" style="display: block; margin: auto;" />
