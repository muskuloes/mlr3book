# Model Interpretation {#interpretation}

In principle, all generic frameworks for model interpretation are applicable on the models fitted with `mlr3` by just extracting the fitted models from the [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) objects.

However, two of the most popular frameworks,

* [iml](https://cran.r-project.org/package=iml) in Subsection \@ref(iml),
* [DALEX](https://cran.r-project.org/package=DALEX) in Subsection \@ref(dalex), and

additionally come with some convenience for `mlr3`.
