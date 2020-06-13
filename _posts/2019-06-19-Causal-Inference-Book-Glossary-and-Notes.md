---
layout: post
comments: true
include: true
title:  "Causal Inference Book Part I -- Glossary and Notes"
excerpt: "Key concepts from Part 1 of Hernan and Robins Causal Inference Book"
date:   2019-06-19 11:50:00
mathjax: true
---

This page contains some notes from Miguel Hernan and Jamie Robin's [Causal Inference Book](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/).  So far, I've only done Part I.

This page only has key terms and concepts. On [this page](https://sgfin.github.io/2019/06/19/Causal-Inference-Book-All-DAGs/), I've tried to systematically present all the DAGs in the same book.  I imagine that one will be more useful going forward, at least for me.


__Table of Contents__:
* TOC
{:toc}

### A few common variables

| Variable | Meaning |
| :-----------: |:-------------:|
| _A_, _E_  | Treatment |
| _Y_ | Outcome |
| _Y^(A=a)_ | Counterfactual outcome under treatment with $$a$$ |
| _Y^(a,e)_ | Joint counterfactual outcome under treatment with $$a$$ and $$e$$ |
| _L_ | Patient variable (often confounder) |
| _U_ | Patient variable (often unmeasured or background variable)  |
| _M_ | Patient variable (often effect modifier) |

### Chapter 1: Definition of Causal Effect

| Term | Notation or Formula | Notes | Page | 
| :-----------: |:-------------:|
|  __Association__ |  Pr[Y=1\|A=1] $$\neq$$ Pr[Y=1\|A=0] | _Example definitions of independence (lack of association)_: <br/> Y $$\unicode{x2AEB}$$ A <br/> or <br/> Pr[Y=1\|A=1] - Pr[Y=1\|A=0] = 0 <br> or <br> $$\frac{Pr[Y=1\|A=1]}{Pr[Y=1\|A=0]}$$ = 1 <br/> or <br/> $$\frac{Pr[Y=1\|A=1]/Pr[Y=0\|A=1]}{Pr[Y=1\|A=0]/Pr[Y=0\|A=0]}$$ = 1  | I.10 |
| __Causation and Causal Effects__ | _Causation_:<br/>Pr[Y^(a=1)=1] $$\neq$$ Pr[Y^(a=0)=1] <br/><br/> _Individual Causal Effects_:<br/>  Y^(a=1) - Y^(a=0) <br/><br/> _Population Average Causal Effects_:<br/> E[Y^(a=1)] - E[Y^(a=0)] <br/><br/> _where_ <br/>Y^(a=1) = Outcome for treatment w/ $$a=1$$ <br/> Y^(a=0) = Outcome for treatment w/ $$a=0$$ | _Sharp causal null hypothesis_: <br/>Y^(a=1) = Y^(a=0) for all individuals in the population. <br/><br/> _Null hypothesis of no average causal effect_: <br/> E[Y^(a=1)] = E[Y^(a=0)] <br/><br/> _Mathematical representations of causal null_: <br/> Pr[Y^(a=1)=1] - Pr[Y^(a=0)=1] = 0 <br> or <br> $$\frac{Pr[Y^{a=1}=1]}{Pr[Y^{a=0}=1]} = 1$$ <br/> or <br/> $$\frac{Pr[Y^{a=1}=1]/Pr[Y^{a=1}=0]}{Pr[Y^{a=0}=1]/Pr[Y^{a=1}=0]} = 1$$  | I.7 |

### Chapter 2: Randomized Experiments

| Term |  Notes | Page | 
| :-----------: |:-------------:|
| __Marginally randomized experiment__ | Single unconditional (marginal) randomization probability applied to assign treatments to all individuals in experiment. <br/><br/> Produces exchangeability of treated and untreated. <br/><br/> Values of counterfactual outcomes are __missing completely at random (MCAR)__. | I.18 |
| __Conditionally randomized experiment__ | Randomized trial where study population is stratified by some variable $$L$$, with different treatment probabilities for each stratum. <br/><br/>  Needn't produce _marginal exchangeability_, but produces _conditional exchangeability_. <br/><br/> Values of counterfactuals are _not_ MCAR, but are __missing at random (MAR)__ conditional on $$L$$. | I.18 |
| __Standardization__ | Calculate the _marginal_ counterfactual risk from a _conditionally randomized experiment_ by taking a weighted average over the stratum-specific risks.  <br/><br/>  Standardized mean:  <br/><br/>  $$\sum_l E[Y\|L=l,A=a] \times Pr[L=l]$$ <br/><br/>  Causal risk ratio can be computed via standardization as follows:  <br/><br/>  $$\frac{Pr[Y^{a=1}=1]}{Pr[Y^{a=0}=1]} = \frac{\sum_l E[Y=1\|L=l,A=1]\times Pr[L=l]}{\sum_l E[Y=1\|L=l,A=1]\times Pr[L=l]}$$ | I.19 |
| __Inverse probability weighting__ | Given a conditionally randomized study population: <br/><img src="/assets/hernan_dags/2_1.png" width="300"> <br/> We can invoke an assumption of conditional exchangeability given $$L$$ to simulate the counterfactual in which everyone had received (or not received) the treatment:  <br/> <img src="/assets/hernan_dags/2_2.png" width="300"> <br/>. The causal effect ratio can then be directly calculated by comparing <br/> $$Pr[Y^{a=1}=1]/Pr[Y^{a=0}=1]$$ (in this example, it's $$\frac{10/20}{10/20}=1$$.) <br/><br/> By the same token, you can effectively double your population and create a hypothetical _pseudo-population_ in which everyone had received both treatments: <br/> <img src="/assets/hernan_dags/2_3.png" width="300"> <br/><br/> This process amounts to weighting each individual in the population by the inverse of the conditional probability of receiving the treatment she received (see formula on right above). Hence the name _inverse probability (IP) weighting_. | I.20 |

### Chapter 3: Observational Studies

| Term | Notation or Formula | English Definition  | Notes | Page | 
| :-----------: |:-------------:|
| __Identifiability conditions__ | See below. | Sufficient conditions for conceptualizing an observational study as a randomized experiment. <br/><br/> Consist of: <br/>  1. Consistency <br/> 2. Exchangeability, and <br/> 3. Positivity. | | I.25 |
|  __Consistency__  | If $$A_i$$ = $$a$$, then $$Y_{i}^{a}=Y^{A_i}$$ = $$Y_i$$ | "The values of treatment under comparison correspond to well-defined interventions that, in turn, correspond to the versions of treatment in the data." <br/><br/> Has two main components: <br> 1. Precise specification of counterfactual outcomes Y^a, and <br> 2. Linkage of counterfactual outcomes to observed outcomes.  | Violated in an ill-defined intervention. <br/><br/> Examples: <br> - Study looks at "heart transplant" but doesn't look at protocols (e.g. which immunosuppresant is used). If effect varies between versions of treatment and protocols not equally distributed, could cause problems. <br/> - Study wants to look at "obesity", but "non-obesity" lumps together non-obesity from exercise vs cachexia vs genes vs diet. Need to subset population or make assumption that specific source of non-obesity doesn't impact outcome. (Assumption called _treatment-variation irrelevance_ assumption.) <br/><br/> Not a testable assumption, relies on domain expertise. | I.31 |
|  __Exchangeability__ (aka exogeneity) | Y^a $$\unicode{x2AEB}$$ A for all $$a$$ <br/><br/> or <br/><br/> Pr[Y^a=1 \| A=1] = Pr[Y^a=1 \| A=0] = Pr[Y^a=1]|  "The treated, had they remained untreated, would have experienced the same average outcome as the untreated did, and vice versa." <br/><br/> Essentially, this is the assumption of no unmeasured confounding.  |   Beware formula: Not the same as Y $$\unicode{x2AEB}$$ A, which would mean treatment has no effect on outcome. | I.27 |
| __Conditional exchangeability__ | Y^a $$\unicode{x2AEB}$$ A \| L for all a <br/><br/> or <br/><br/> Pr[Y^a=1 \| A=1, L=1] = Pr[Y^a=1 \| A=0, L=1] = Pr[Y^a=1] \| L=1 | "The conditional probability of receiving every value of treatment is randomized or depends only on measured covariates" | Think conditional RCT where assigment depends only on $$L$$. <br/><br/> In observational studies, this is an untestable assumption, thus relies on domain expertise. | I.27 |
| __Positivity__ | Pr[A=a \| L=$$l$$ ] > 0 for all values $$l$$ with Pr[L=$$l$$] $$\neq$$ 0 in the population of interest | "The conditional probability of receiving every value of treatment is greater than zero, i.e. positive." | Aka "Experimental treatment assumption" <br/><br/> Example of positivity not holding: doctors always give heart transplants to patients in critical condition, eliminitating positivity from that stratum of an observational study. <br/><br/> Unlike exchangeability, positivity, _can_ be empricially verified.| I.30 |

### Chapter 4: Effect Modification

| Term | Notation or Formula | English Definition  | Notes | Page | 
| :-----------: |:-------------:|
| __Effect modification__ <br/> aka effect-measure modification | _Additive effect modification_: <br/> E[Y^(a=1)-Y^(a=0) \| M = 1] $$\neq$$ E[Y^(a=1)-Y^(a=0) \| M = 0] <br/><br/> _Multiplicative effect modification_: <br/>  $$\frac{E[Y^{a=1} \| M = 1]}{E[Y^{a=0} \| M = 1]}$$ $$\neq$$ $$\frac{E[Y^{a=1}\| M = 0]}{E[Y^{a=0}\| M = 0]}$$ |  $$M$$ is a modifier of the effect of $$A$$ on $$Y$$ when the average causal effect of $$A$$ on $$Y$$ varies across levels of $$M$$. | The _null hypothesis of no average causal effect_ does *not* necessarily imply the absence of effect modification (e.g. equal and oppositive effect modifications in men and women could cancel at the population level), but the _sharp null hypothesis of no causal effect_ does imply no effect modicifaction. <br/><br/>  We only count variables _unaffected by treatment_ as effect modifiers. Similar variables that are effected by treatment are termed __mediators__. | I.41 |
| __Qualitative effect modification__ | | Average causal effects in different subsets of the population go in opposite directions.   | In presence of qualitative effect modification, additive effect modification implies multiplicative effect modification, and vice versa.  In absence of qualitative effect modification, it's possible to have only additive or only multiplicative effect modification. <br/><br/> Effect modifiers are not necessarily assumed to play a causal role. To make this explicit, sometimes the terms _surrogate effect modifier_ vs _causal effect modifier_ are used, or you can play it even safer and refer to "effect heterogeneity across strata of $$M$$." <br/><br/> Effect modification is helpful, among other things, for (i) assessing transportability to new populations where $$M$$ may have different prevalences, (ii) choosing subpopulations that may most benefit from treatment, and (iii) identifying mechanisms leading to outcome if modifiers are mechanistically meaningful (e.g. circumscision for HIV transmission).  | I.42 |
| __Stratification__   | _Statified causal risk differences_: <br/> E[Y^(a=1) \| M = 1] - <br/> E[Y^(a=0) \| M = 1] <br/>and<br/> E[Y^(a=1) \| M = 0] - <br/> E[Y^(a=0) \| M = 0] | To _identify_ effect modification by variable $$M$$, separately compute the causal effect of $$A$$ on $$Y$$ for each statum of the variable $$M$$.  | If study design assumes conditional rather than marginal exchangeability, analysis to estimate effect modification must account for all other variables $$L$$ required to give exchangeability. This could involve standardization (IP weighting, etc.) by $$L$$ within each stratum $$M$$, or just using finer-grained stratification over all pairwise combinations of $$M$$ and $$L$$ (see page I.49). <br/><br/> By the same token, stratification can be an alternative to standardization techinques such as IP weighting in analysis of any conditional randomized experiment : instead of estimating an average causal effect over the population while standardizing for $$L$$, just stratify on $$L$$ and report separate causal effect estimates for each stratum. | I.43-49|
| __Collapsibility__ | | A characteristic of a population _effect measure_. Means that the effect measure can be expressed as a weighted average of stratum-specific measures. |  Examples of collapsible effect measures: risk ratio and risk difference <br/><br/> Example of non-collapsible effect measure: odds ratio. <br/><br/> Noncollapsibility can produce counter-intuitive findings like a causal odds ratio that's smaller in the average population than in any stratum of the population.  | I.53 | 
| __Matching__ | | Construct a subset of the population in which all variables $$L$$ have the same distribution in both the treated and the untreated. | Under assumption of conditional exchangeability given $$L$$ in the source population, a matched population will have unconditional exchangeability. <br/><br/>  Usually, constructed by including all of the smaller group (e.g. the treated) and selecting one member of the larger group (e.g. the untreated) with matching $$L$$ for each member in the smaller group. Often requires approximate matching. | I.49 |
| __Interference__ | | Treatment of one individual effects treatment status of other individuals in the population. | Example: A socially active individual convinces friends to join him while exercising.   | I.48 |
| __Transportability__ | | Ability to use causal effect estimation from one population in order to inform decisions in another ("target") population.  <br/><br/>  | Requires that the target population is characterized by comparable patterns of: <br/> - Effect modification <br/> - Interference, and <br/> - Versions of treatment |  I.48 |

### Chapter 5: Interaction

| Term | Notation or Formula | English Definition  | Notes | Page | 
| :-----------: |:-------------:|
| __Joint counterfactual__ | Y^(a,e) | Counterfactual outcome that would have been observed if we had intervented to set the individual's values of $$A$$ (treatment component 1) to $$a$$ and $$E$$ (treatment component 2) to $$e$$. | | I.55 | 
| __Interaction__ | _Interaction on the additive scale_: <br/> Pr[Y^(a=1,e=1)=1] - Pr[Y^(a=0,e=1)=1] $$\neq$$ Pr[Y^(a=1,e=0)=1] - Pr[Y^(a=0,e=0)=1] <br/> <br/> or <br/> <br/> Pr[Y^(a=1) = 1 \| E=1 ] - Pr[Y^(a=0) = 1 \| E=1 ] $$\neq$$ Pr[Y^(a=1) = 1 \| E=0 ] - Pr[Y^(a=0) = 1 \| E=0] | The causal effect of $$A$$ on $$Y$$ after a joint intervention that set $$E$$ to 1 differs from the causal effect of $$A$$ on $$Y$$ after a joint intervention that set $$E$$ to 0. (Definition also holds if you swap $$A$$ and $$E$$.) | Different from effect modification because an effect modifier $$M$$ is not considered a treatment or otherwise a variable on which we can intervene.  In interaction, interventions $$A$$ and $$E$$ have equal status. <br/><br/> Note from definition 2 on the left, however, that the mathematical definitions of effect modification and interaction line up. This means that if you _randomize_ an interactor, it becomes equivalent to an effect modifier. <br/><br/> Inference over joint counterfactuals require that the identifying conditions of exchangeability, positivity, and consistency hold for _both_ treatments.  | I.55|
| __Counterfactual response type__ | | A characteristic of an _individual_ that refers to how she will respond to a treatment. | For example, an individual may have the same counterfactual outcome regardless of treatment, be helped by the treatment, or be hurt by the treatment. <br/><br/>  The presence of an interaction between $$A$$ and $$E$$ implies that some individuals exist such that their counterfactual outcomes under $$A=a$$ cannot be determined without knowledge of $$E$$. | I.58 | 
| __Sufficient-component causes__ | | A set of variables that are sufficient to determine the outcome for a specific individual. |  The minimal set of sufficient causes can be different for distinct ndividuals in the same study. For example, a patient with background factor $$U_1$$ might have the same outcome regardless of treatment, whereas another patient's outcome might be driven by both a treatment $$A$$ and interactor $$E$$. <br/><br/> Minimal sufficient-component causes are sometimes visualized with pie charts. <br/><br/> _Contrast between counterfactual outcomes framework and sufficient-component-cause framework:_ <br/> _Sufficient outcomes framework_ focuses on questions like: "given a particular effect, what are the various events which might have been its cause?" and _counterfactual outcomes framework_ focuses on questions like: "what would have occurred if a particular factor were intervened upon and set to a different level than it was?".  Sufficient-component-causes requires more detailed mechanistic knoweldge and is generally more a pedagological tool than a data analysis tool. | I.61 |
| __Sufficient cause interaction__ | | A sufficient cause interaction between $$A$$ and $$E$$ exists in a population if $$A$$ and $$E$$ occur together in a sufficient cause. | Can be _synergistic_ (A = 1 and E = 1 present in sufficient cause)  or _antagonistic_ (e.g. A = 1 and E = 0 is present in sufficient cause) .  | I.64 |



### Chapter 6:  Causal Diagrams 

| Term | Definition | Page | 
| :-----------: |:-------------:|
| __Path__ | A path on a DAG is a sequence of edges connecting two variables on the graph, with each edge occurring only once.  | I.76 |
| __Collider__ | A collider is a variable in which two arrowheads on a path collide.  <br/> <br/> For example, $$Y$$ is a collider in the path $$A \rightarrow Y \leftarrow L$$ in the following DAG: <br/> <img src="/assets/hernan_dags/6_1.png" width="150"> | I.76 |
| __Blocked path__ | A path on a DAG is blocked if and only if: <br/>1. it contains a noncollider that has been conditioned, or <br/> 2. it contains a collider that has not been conditioned on and has no descendants that have been conditioned on.  | I.76 |
| __d-separation__ | Two variables are d-separated if all paths between them are blocked | I.76 |
| __d-connectedness__ | Two variables are d-connected if they are not d-separated | I.76 |
| __Faithfulness__ | Faithulness is when all non-null associations implied by a causal diagram exist in the true causal DAG. Unfaithfulness can arise, for example, in certain settings of effect modification, by design as in matching experiments, or in the presence of certain deterministic relations between variables in the graph.    | I.77 |
| __Positivity__ (on graphs) | The arrows from the nodes $$L$$ to the treatment node $$A$$ are not deterministic. (Concerned with nodes _into_ treatment nodes) | I.75 |
| __Consistency__ (on graphs) | Well-defined intervention criteria:  the arrow from treatment $$A$$ to outcome $$Y$$ corresponds to a potentially hypothetical but relatively unambiguous intervention. (Concerned with nodes _leaving_ the treatment nodes.) | I.75 |
| __Systematic bias__ | The data are insuffient to identify the causal effect even withan infinite sample size. This occurs when any sturctural association between treatment and outcome does not arise from the causal effect of treatment on outcome in the population of interest. | I.79 | 
| __Conditional bias__ | _For average causal effects within levels of $$L$$_: <br/> Conditional bias exists whenever the effect measure (e.g. causal risk ratio) and the corresponding association measure (e.g. associational risk ratio) are not equal.<br/> Mathematically, this is when: <br/> $$Pr[Y^{a=1} \| L = l] - Pr[Y^{a=0} \| L = l]$$ differs from $$Pr[Y\|L=l, A = 1] - Pr[Y\|L-l, A=0]$$ for at least one stratum $$l$$.<br/><br/> _For average causal effects in the entire population_: <br/> Conditional bias exists whenever <br/> $$Pr[Y^{a=1} ] - Pr[Y^{a=0}]$$ $$\neq$$ $$Pr[Y = 1\| A = 1] - Pr[Y = 1 \| A = 0]$$.  | I.79 |
| __Bias under the null__ | When the null hypothesis of no causal effect of treatment on the outcome holds, but treatment and outcome are associated in the data. <br/><br/>Can be from either confounding, selection bias, or measurement error..  | I.79 |
| __Confounding__ | The treatment and outcome share a common cause.  | I.79 |
| __Selection bias__ | Conditioning on common effects.  | I.79 |
| __Surrogate effect modifier__| An effect modifier that does not dirrectly influence that outcome but might stand in for a __causal effect modifier__ that does. | I.81 | 


### Chapter 7:  Confounding

| Concept | Definition or Notes| Page | 
| :-----------: |:-------------:|
| __Backdoor Path__ | A noncausal path between treatment and outcome that remains even if all arrows pointing from treatment to other variables (the descendants of treatment) are removed. That is, the path has an arrow pointing into treatment.  | I.83 |
| __Confounding by indication__ (or __Channeling__) | A drug is more likely to be prescribed to individuals with a certain condition that is both an indication for treatment and a risk factor for the disease.  |  I.84  |
| __Channeling__ | Confounding by indication in which patient-specific risk factors $$L$$ encourage doctors to use certain drug $$A$$ within a class of drugs.  |  I.84  |
| __Backdoor Criterion__ | Assuming consistency and positivity, the _backdoor criterion_ sets the circumstances under which (a) confounding can be eliminated from the analysis, and (b) a causal effect of treatment on outcome can be identified.  <br/><br/>  Criterion is that identifiability exists if all backdoor paths can be blocked by conditioning on variables that are not affected by the treatment.  <br/><br/> The two settings in which this is possible are: <br/><br/>  1. No common causes of treatment and outcome. <br/><br/> 2. Common causes but enough measured variables to block all colliders.   | I.85 |
| __Single-world intervention graphs (SWIG)__ | A causal diagram that unifies counterfactual and graphical approaches by explicitly including the counterfactual variables on the graph. <br/><br/>  Depicts variables and causal relations that would be observed in a hypothetical world in which all individuals received treatment level $$a$$. In other words, is a _graph_ that represents the counterfactual _world_ created by a _single intervention_, unlike normal DAGs that represent variables and causal relations from the actual world. | I.91 |
| __Two categories of methods for confounding adjustment__  | __G-Methods__:<br/> G-formula, IP weighting, G-estimation. Exploit conditional exchangeability in subsets defined by $$L$$ to estimate the causal effect of $$A$$ on $$Y$$ in the entire population or in any subset of the population. <br/><br/> __Stratification-based Methods__: Stratification, Restriction, Matching.  Methods that exploit conditional exchangeability in subsets defined by $$L$$ to estimate the association between $$A$$ and $$Y$$ in those subsets only. | I.93 |
| __Difference-in-differences__ and __negative outcome controls__ | A technique to account for unmeasured confounders under specific conditions. <br/><br/>  The idea is to measure a "negative outcome control", which is the same as the main outcome but _right before treatment_.  Then, instead of just reporting the effect of the treatment on the _outcome_ (treatment effect + confounding effect), you substract out the effect of treatment on the _negative outcome_ (only confounding effect). What's left is is the _difference-in-differences_. <br/><br/> This requires the assumption of _additive equi-confounding_: <br/> $$E[Y^{0}\|A=1] - E[Y^{0}\|A=0]$$ = $$E[C\|A=1] - E[C\|A=0]$$. <br/><br/> Negative outcome controls are also sometimes used to try to _detect_ confounding. <br/><br/> Note: The DAG demonstration (Figure 7.11) is really useful for this one. | I.95 |
| __Frontdoor criterion__ and __Frontdoor adjustment__ | A two-step standardization process to estimate a causal effect in the presence of a confounded causal effect _that is mediated by an unconfounded mediator variable_. <br/><br/>  Given a DAG such as: <br/> <img src="/assets/hernan_dags/7_12.png" width="150"> <br/> $$Pr[Y^{a}=1] = \sum_{m}Pr[M^{a}=m]Pr[Y^{m}=1]$$. <br/><br/>  Thus, standardization can be applied in two steps: <br/><br/> 1. Compute $$Pr[M^{a}=m]$$ as $$Pr[M=m\| A=a]$$, and <br/><br/> 2. Compute $$Pr[Y^{a}=1]$$ as $$\sum_{a'}Pr[Y=1\|M=m,A=a']Pr[A=a']$$ <br/><br/> These are then combined with the formula <br/>  $$\sum_{m}Pr[M=m\| A=a]\sum_{a'}Pr[Y=1\|M=m,A=a']Pr[A=a']$$ <br/><br/> The name _frontdoor adjustment_ comes because it relies on the path from $$A$$ and $$Y$$ moving through a descendent $$M$$ of $$A$$ that causes $$Y$$.| I.96 |

### Chapter 8:  Selection Bias

__Note__:  I have almost no notes in here, because the DAG section contains pretty much all the content I'm interested in noting here.

| Concept | Definition or Notes| Page | 
| :-----------: |:-------------:|
| __Competing Event__ | An event that prevents the outcome of interest from happening. For example, death is a competing event, because once it occurs, no other outcome is possible. | I.108 |
| __Multiplicative survival model__ | A multiplicative survival model is of the form: <br/><br/> $$Pr[Y=0\|E=e,A=a]=g(e)h(a)$$ <br/><br/> . The data forllow such a model when there is no interaction between $$A$$ and $$E$$ on a multiplicative scale.  This allows, for example, $$A$$ and $$E$$ to be conditionally independent given $$Y=0$$ but not conditionally dependent when $$Y=1$$. See Technical Point 8.2 and the example in Figure 8.13. | I.109 |
| __Healthy worker bias__ | Example of selection bias where people are only included in the study if they are healthy enough, say, to come into work and be tested.  | I.99 |
| __Self-selection bias__ | Example of selection bias where people volunteer for enrollment.  | I.100 |
 
### Chapter 9:  Measurement Bias

| Concept | Definition or Notes| Page | 
| :-----------: |:-------------:|
| __Measurement bias__ or __Information bias__ | Systematic difference in associational risk and causal risk that arises due to measurement error. Eliminates causal inference even under identifiability conditions of exchangeability, positivity, and consistency.  | I.112 |
| __ Independent measurement error __ | Independent measurement error takes place when the measurement error of the treatment ($$U_{A}$$) and the measurement error of the response ($$U_{Y}$$) are d-separated.  Dependent measurement error is when they are d-connected. | I.11 |
| __ Nondifferential measurement error __ | Measurement error is _nondifferential_ with respect to the outcome if $$U_{A}$$ and $$Y$$ are d-separated.  Measurement error is nondifferential with respect to the treatment if $$U_{Y}$$ and $$A$$ are d-separated.  | I.11 |
| __Intention-to-treat effect__ | The causal effect of randomized treatment assigment $$Z$$ in an intention-to-treat trial on the outcome $$Y$$. Depends on the strength of the effect of assignment treatment on outcome ($$Z \rightarrow Y$$), the assignment treatment on actual treatment received ($$Z \rightarrow A$$), and the effect of the actual treatment received on outcome ($$A \rightarrow Y$$). In theory, this does not require adjustment for confounding, has null preservation, and is conservative. See below for comments on latter two. | I.116 |
| __The exclusion restriction__ | (The goal of double-blinding). The assumption that there is no direct arrow from assigned treatment $$Z$$ to outcome $$Y$$ in an intention-to-treat design. | I.117 |
| __Null Preservation__ in an IIT | If treatment $$A$$ has a null effect on $$Y$$, then assigned treatment $$Z$$ also has a null effect on $$Y$$.  Ensure, in theory, that a null effect will be declared when none exists. However, it requires that the exclusion restriction holds, which breaks down unless their is perfect double-blinding.  | I.119 |
| __Conservatism of the IIT vs Per-protocol__ | The IIT effect is supposed to be closer to the null than the value of the per-protocol effect, because imperfect adherence results in attenuation rather than exaggeration of effect. Thus IIT appears to be a lower bound for per-protocol effect (and is thus conservative). <br/><br/> However, there are three issues with this: <br/> 1. Argument assumes monotonicity of effects (treatment same direction for all patients). If, say, there is inconsistent adherence and thus inconsistent effects, then this could become anti-conservative.  <br/> 2. Even given monotonicity, IIT would only be conservative compared to _placebos_, not necessarily head-to-head trials, where adherence in the second drug might be different. <br/> 3. Even if IIT is conservative, this makes it dangerous when goal is evaluating safety, where you arguably want to be more _aggresive_ in finding signal.| |
| __Per-protocol effect__ | The causal effect of randomized treatment that would have been observed if all individuals had adhered to their assigned treatment as specified in the protocol of the experiment.  _Requires adjustment for confounding_. | I.116 |
| __As-treated analysis__ | An analysis to assess for per-protocol effect. Includes _all patients_ and compares those treated ($$A=1$$) vs not treated ($$A=0$$), independent of their assignment $$Z$$. Confounded. | I.118|
| __Conventional per-protocol analysis__ | An analysis to assess for per-protocol effect. Limits the population to those who adhered to the study protocol, subsetting to those for whom $$A=Z$$. Induces a _selection bias_ on $$A=Z$$, and thus still requires adjustment on $$L$$.  | I.118|
| __Tradeoff between ITT and Per-protocol__ | Summary: Estimating the per-protocol effect adds unmeasured confounding, which needs to be (imperfectly) adjusted for. Intention-to-treat adds a misclassification bias, and does not necessarily deliver on purported guarantees of conservatism. As such, there is a real tradeoff, here. | I.117-I.120 |


### Chapter 10:  Random Variability

Sorry, I'm skipping this section, because the key terms are all stats concepts and its mostly a pump-up chapter for the rest of the book.

### Chapter 11: Why Model?

| Concept | Definition or Notes| Page | 
| :-----------: |:-------------:|
| __Saturated Models__ | Models that do not impose restrctions on the data distribution. Generally, these are models whose number of parameters in a conditional mean model is equal to the number of means. For example, a linear model E[ y | x] ~ b0 + b1*x when the population is stratified into only two groups. These are _non-parametric models_. | II.143 |
| __Non-parametric estimator__ | Estimators that produce estimates from the data without any a priori restrictions on the true function. When using the entire population rather than a sample, these yield the true value of the population parameter. | II.143 |

### Chapter 12: 

| Concept | Definition or Notes| Page | 
| :-----------: |:-------------:|
| __Stabilized Weights__ |   |  |
| __Marginal Structure Model__ |   |  |



To-do:
| Concept | Formula | Code | Notes |
| :-----------: |:-------------:|
| __IP Weighting__ |   |  |
| __Standardized IP Weighting__ |   |  |
| __Marginal Structure model__ |   |  |

DONT MISS THE DOUBLY ROBUST ESTIMATOR in TECHNICAL POINT 13.2

