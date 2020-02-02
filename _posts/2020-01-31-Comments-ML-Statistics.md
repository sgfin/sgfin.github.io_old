---
layout: post
comments: false
include: false
title:  'Comments on ML "versus" statistics'
excerpt: ""
date:   2020-01-31 10:00:00
mathjax: false
---

**Why am I writing this?**

Over the last few years, I've observed many vigorous debates about "machine learning versus statistics." Often, these are sparked by some paper/blog post/press release that either (a) involves some use of logistic regression (or some other type of GLM) being described as machine learning, or (b) performs a meta-analysis attempting to pit the fields against each other.

I have allowed myself to get pulled down this rabbit hole far too many times, wasting hours of my time in fruitless debate. As such, I have decided to write this post as a way to inoculate myself against the urge to enter future discussions. The first two sections explain why I consider most ML "versus" Stats debates to be fundamentally flawed, even in their very premise. The following two sections explain why I _do_ validate where people are coming from in having these debates, but still think they (the debates!) are a colossal waste of time.

As time goes on, I might even write a bot to post this on relevant twitter threads. If I do, I will _intentionally_ code this bot using logistic regression and call it machine learning, just to maximize peskiness.

**Outline**
- [Neglected historical context: The term "machine learning" was not coined to contrast with statistics, but to contrast the field with competing paradigms for building intelligent computer systems.](#Sect1)
- [Arguments about who "owns" regression miss the point](#Sect2)
- [Distinctions in goals have yielded divergence in methods and cultures, which explains shifting connotations of the term "machine learning"](#Sect3)
- [Isn't this whole "debate" a massive waste of time?](#Sect4)



### <a name="Sect1"></a> Neglected historical context: The term "machine learning" was _not_ coined to contrast with statistics, but to contrast the field with competing paradigms for building intelligent computer systems.

Before getting to Machine Learning (ML), a couple paragraphs on Artificial Intelligence (AI). These days, many people -- including me -- reflexively wince when they hear the term "AI," because it is (a) used by slimey buzzword peddlers to such an extent that it is now nearly synonymous with "snakeoil," (b) overloaded with connotations of sentient killer robots, and (c) almost exclusively used to refer to machine learning, anyway. This is all quite unfortunate. However, try to set that aside for just one paragraph.

Engineers have dreamed of building something "smart" for thousands of years, but the term "artificial intelligence" itself was coined by John McCarthy in preparation for the famous "Dartmouth Conference" of 1956. McCarthy defined artificial intelligence as "the science and engineering of making intelligent machines" and that's not too bad for a pithy one-liner. Importantly, he was able to convince his colleagues to adopt this term at Dartmouth in large part because it was _vague_.  At that point in time, computer scientists who were trying to crack intelligence were focused _not_ on data-driven methods, but on things like automata theory, formal logic, and cybernetics. McCarthy wanted to create a term that would capture all of these paradigms (and other ones yet to come) rather than favoring any specific approach.

It was with _this_ context that Arthur Samuel (one of the attendees at the Dartmouth Conference) coined the term "Machine Learning" in 1959, which he defined as:

> Field of study that gives computers the ability to learn without being explicitly programmed.

Samuels and his colleagues wanted to help computers becomes "smart" by equipping them with the capacity to recognize patterns and iteratively improve over time. While this may seem like an obvious approach today, it took decades before this became the dominant mode of AI research (as opposed to, say, building systems that exhibit "intelligence" by applying propositional logic over curated knowledge graphs).

In other words, machine learning was coined to describe a design process for computers that _leverages_ statistical methods to improve performance over time. The term was created by computer scientists, for computer scientists, and designed as a contrast with _non-data-driven approaches_ to building smart machines. It was _not_ designed to be a contrast with _statistics_, which is focused on using (often overlapping) data-driven methods to inform humans.

Another extremely widely-referenced definition of ML comes from Tom M. Mitchell's 1997 textbook, which said:

> The field of machine learning is concerned with the question of how to construct computer programs that automatically improve with experience,

and offered the accompanying semi-formal definition:

> A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P, improves with experience E.

This is all very much in accordance with Arthur Samuel's definition, and I could pull other more recent definitions with similar verbiage. Another passage from Mitchell that I think gets less circulation than it deserves, however, is the following (taken with a little reformatting from a 2006 article called ["The Discipline of Machine Learning](http://www.cs.cmu.edu/~tom/pubs/MachineLearning.pdf)"):

> Machine Learning is a natural outgrowth of the intersection of Computer Science and Statistics. We might say the defining question of Computer Science is "How can we build machines that solve problems, and which problems are inherently tractable/intractable?" The question that largely defines Statistics is "What can be inferred from data plus a set of modeling assumptions, with what reliability?"
> 
> The defining question for Machine Learning builds on both, but it is a distinct question. Whereas Computer Science has focused primarily on how to manually program computers, Machine Learning focuses on the question of how to get computers to program themselves (from experience plus some initial structure). Whereas Statistics has focused primarily on what conclusions can be inferred from data, Machine Learning incorporates additional questions about what computational architectures and algorithms can be used to most effectively capture, store, index, retrieve and merge these data, how multiple learning subtasks can be orchestrated in a larger system, and questions of computational tractability.


### <a name="Sect2"></a> Arguments about who "owns" regression miss the point

Given the history just described, I must admit that I've been frustrated at times by the authoritarian tone with which many have tried to enforce a false dichotomization between statistics and ML methods. In particular, there appears to be a strange fixation by specific online personalities on insisting that regression-powered applications must _not_ be described as ML. In reading some of these discussions, one might even come away thinking that there is a conspiracy at play to annex regression away from statistics.

To me, this is only slightly less silly than trying to stir up a turf war to target anyone who uses logistic regression and calls it "econometrics." For at least 60 years, "machine learning" has been about building the best learning computers that we can, not some weird methods competition with statistics. This is why, when it comes to teaching "ML methods," almost every introductory machine learning textbook or course that I've ever seen -- Mitchell, Murphy, Bishop, Ng, etc. --  spends much of its efforts on teaching GLMs and their variants. It's also why it's _perfectly sensible_ for specialized textbooks to include plots like [this one](https://images.app.goo.gl/WxjjxjUXSxQw75To7), which routinely makes the rounds to much pearl-clutching. Would I expect such a plot to be useful to statisticians? Or course not!  But they make sense in context of the ML/AI fields, which are concerned with different ways to make programs act "smart".  And to put it bluntly for any [c]rank-y colleagues in statistics: _You don't get to decide_ which taxonomies another field finds useful in framing its own problems and history.

The great irony with the whole recurring snafoo around who "owns" regression -- and all of its variants -- is that it simultaneously undersells _both_ machine learning _and_ statistics, for many reasons that include the following four:

- First, it minimizes -- or even defines away -- the core role that classic statistical methods _continue_ to play in efforts to build computer programs that learn.
- Second, it ignores the impact that ML has had on statistics, when in reality AI and CS have been a _massive_ boon to statistics research. This includes the generation of new statistical paradigms (e.g. Judea Pearl and others' work on causality, now one of many booming subareas of stats that came from ML) and a wide array of algorithmic and computational tools that have enabled the rise of statistical computing.
- Third, a false dichotomy between ML and stats minimizes the wide -- and critical to modeling decisions -- variation _within_ each purported class. It's also silly. Consider two examples:
  - First, consider the following Schroedinger's "am I doing stats research or ML research" experiment: Implement a 1-layer, fully connected neural net in Pytorch. Now, add a random variable that, when equal to one, toggles on a second layer -- you now have a quantum state of stats research and "profoundly different" ML research!  Now instead, add a thousand super crazy complicated interaction terms and an elastic net penalty -- meh, a trivial difference! Now, form a research team to integrate it into an autonomous car -- meh, still the same vanilla stats research!  Etc. (Note:  Before you rag me too hard for this one, be sure to read the next section.)
  - Second, consider a meta-analysis that attempts to compare "ML performance" against that of "statistical models," but a large chunk of the ML camp is comprised of papers using (single!) decision trees. What a useless comparison in 2020!  This is not unlike pitting China against the U.S. in some sporting event, but then having most of the team from one country be comprised of its 6-and-under little league squad. Same deal if you have the regression analyses being done without the care of a professional statistician. The spectrum is so broad (and the execution so essential) within each of these purportedly distinct methodological camps that most statements about the whole collections are minimally helpful. It's all about picking the right tool for a specific job, and then taking the time and effort to use that tool properly!
- Four, the above dispute also ignores the fact that many top researchers, publication venues, and papers in stats or ML are fully-fledged citizens of both communities.

In my opinion, the careers of Trevor Hastie and Rob Tibshirani highlight the best of what happens when statisticians interact richly with machine learning researchers. Rather than getting caught up in drawing methodological border lines, they have taken tools developed first by machine learning researchers and helped formally situate them _within_ the world of statistics proper. In this light, I enjoy their frequent use of the term "statistical learning" (as in the title of their textbook), which I think nicely emphasizes the fact that their _goals_ are those of statistics, even if many of the _methods_ in the book have been developed by and for people in ML. (I'll also point out, a bit immaturely, that I've never heard a machine learning researcher complain that Hastie and Tibshirani are trying to annex their methods by not using the phrase "machine learning" when describing neural networks, tree-based methods, etc. Nor have I heard complaints about who "owns" computational statistics, even though its very existence is built on 20th century CS.)

All the above being said, I do appreciate that perfectly reasonable people have come to think of ML as a disjoint set of _methods_ from statistics. The following sections elaborate on why I think this has happened, and what I think this means as a takeaway for the overall discussion.


### <a name="Sect3"></a> Distinctions in goals have yielded a divergence in methods and cultures, which explains shifting connotations of the term "machine learning."  Thus many "debates" are doomed to futility before they begin.

As stated above, the field of machine learning research was founded as computer scientists sought to build and understand intelligent computer systems, and this continues to be the case today. Major ML applications include things like speech recognition, computer vision, robotics/autonomous systems, computational advertising (sigh...), surveillance (sigh...), chat-bots (sigh...), etc. In seeking to solve these problems, machine learning researchers will almost _always_ start by first trying classical statistical methods, including the relevant simple GLM (in fact, this is often considered a mandatory baseline for publication in many applied ML areas). Hence my whole discussion about ML not being predicated on a specific method. _However_, computer scientists have, of course, also significantly added to this toolkit over the years through the development of additional methods.

As with evolution in any other context, the growing phylogeny of statistical methods used for machine learning have been shaped by selective pressures. Compared to statisticians, machine learning researchers typically care much less about _understanding_ any specific action taken by their algorithms (though it is certainly important, and increasingly a bigger priority). Rather, they usually care most about minimizing _model errors_ on held-out data. As such, it makes sense that methods developed by ML researchers are typically more flexible even at the expense of interpretability, for example. [Leo Breiman](https://projecteuclid.org/euclid.ss/1009213726) and others have written about how these cultures have informed methods development, such as random forests. This often-divergent evolution has made it easy to draw (fuzzy) boundaries between ML and statistics research based entirely on _methods_. To boot, many statisticians are unaware of the history of ML, and have thus, for years, only ever been exposed to the field by means of the methods it periodically emits. It is thus unsurprising that they would be interested in defining the field in any other terms, even if it is dissapointing.

By the same token, a sharp division based on _use_ (like I advocated for above) is now complicated by the fact that many ML people say they're doing machine learning even when they're applying their methods for pure data analysis rather than to drive a computer program. While arguably incorrect in a strict historical sense, I don't fault people for doing this -- probably out of a mixture of habit, cultural affiliation, and/or because it sounds cool.

Taken together, people now use "machine learning" to mean very different things. Sometimes, people use it to mean: "I'm using a statistical method to make my program learn" or "I'm developing a data analysis that I hope to deploy in an automated system." Other times, they mean: "I'm using a method -- perhaps for a statistical data analysis -- that was originally developed by the machine learning community, like random forests."  Still other times (maybe most of the time…?), they mean: "I consider myself a machine learning researcher, I'm working with data, and I can call this work whatever I darn well please."

These different uses of the term aren't really surprising or problematic, because this is simply how language evolves. But it does make it extremely frustrating when a hoard of data scientists (oh no, another hypey term! I use it here as union of ML and statistics) collectively try to debate whether or not a specific project can be branded as ML or must simply be branded "just statistics." Usually, when this happens, people enter the discussion with wildly different assumptions -- poorly defined, and seldom articulated -- about what the words mean in the first place. And then they rarely take the time to understand where others are coming from or what they are actually trying to say. Instead, they typically just talk past each other, louder instead of clearer.


### <a name="Sect4"></a> Isn't this whole "debate" a massive waste of time?

Finally, let's lay our cards on the table:  There are plenty  of machine learning researchers (or at the very least, machine learning hobbyists), who exhibit an inadequate understanding of statistics for people who work with data for a living. At times, I _am_ such a machine learning researcher! (Though I'd wager that many professional statisticians sometimes feel the same way, too.) More seriously, ML research moves so fast, and is sometimes so culturally disconnected from the field of statistics, that I think that it is all-too-common for ML practitioners to re-discover or re-invent parts of statistics. That's a problem and a waste.

That said, I feel that the solution to both of these problems is to increase recognition that ML's data methods live _within_ statistics. Rather than doubling down on a false partition between the two fields, our priority needs to be the cultivation of a robust understanding of statistical principles, whether they are being used for data analysis or for programming intelligent systems. Endless debates about what to _call_ a lot of this work ends up distracting people from essential conversations about how to _carry out_ good work by matching the right **specific** tool to the right problem. If anything, a fixation on a false dichotomy between stats and ML probably drives many people _further_ into the habit of using unnecessarily complex methods, just in order to feel (whether for pride or for money) like they are doing "real ML" (whatever on earth that means).

Finally, this golden age of statistical computing is driving these two fields closer than ever. ML research, of course, lives within computer science, and the modern statistician is increasingly dependent upon the algorithms and software stack that have been pioneered by CS departments for decades. Modern statisticians -- especially in fields like computational biology -- are also increasingly finding use for methods pioneered by ML researchers for, say, regression in high dimensions or at large scale. On the flip side, the ML community is becoming increasingly concerned with topics like interpretability, fairness, certifiable robustness, etc., which is leading many researchers' priorities to align more directly with the traditional values of statistics. At the very least, even when a system is deployed using the most convoluted architectures possible, it's pretty universally recognized that classical statistics is necessary to measure and evaluate performance.

### In summary:

The whole debate is misguided, the terms are overloaded, the methodological dichotomy is false, ML people care (and increasingly so) about statistics, stats people are increasingly dependent upon CS and ML, and there is no regression annexation conspiracy. There's a lot of hype out there right now, but that doesn't change the fact that, _often_, when people use different terminology than you, that's because they come from a different background or have different goals in mind, not because they are stupid or dishonest. Let's just all be friends and strive to do good work together and learn from each other. Kumbaya.
