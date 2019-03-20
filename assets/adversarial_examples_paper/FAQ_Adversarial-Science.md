---
layout: page
comments: false
title:  "FAQ on Medical Adversarial Attacks Policy Piece"
excerpt: ""
date:   2019-03-21
mathjax: false
---

* TOC
{:toc}


### Why this FAQ?

Last Spring, some colleages (chiefly Andy Beam) and I released a [preprint](https://arxiv.org/pdf/1804.05296.pdf) on adversarial attacks on medical computer visions systems. This manuscript was targeted at a technical audience, and written with the goal of explaining why adversarial attacks researchers should consider healthcare applications and providing some proofs of concept.  I ended up getting a lot of great feedback/pushback via email and twitter, which I really appreciated and which informed an update of the preprint.

After the article was released, we were also put in touch with [Jonathan Zittrain](https://hls.harvard.edu/faculty/directory/10992/Zittrain) and John Bowers from Harvard Law School as well as [Joi Ito](https://www.media.mit.edu/people/joi/overview/) of the MIT Media Lab. These are incredibly thoughtful people with a lot of amazing experience. We decided to write a follow-up article targeted more at medical and policy folks, with the intention of examining precedence for adversarial attacks in the healthcare system as it exists today and initiating a conversation about what to do about them going forward. The result is being published today in Science, here. It's been an absolutely pleasure working with these guys.

Given the nature of the topic, I've been fretting quite a bit that the paper will be misconstrued. At a minimum, I anticipate getting a lot of the same questions I got the first time around on the preprint, and figured it'd be easier to write up answers to these in one place.



### Do you really think adversarial attacks are the biggest concern in using machine learning in healthcare? (A: No!) Then why write the paper?

Adversarial attacks consitute just one small part of a large taxonomy of potential pitfalls of machine learning (both ML in general and medical ML in particular).

When I think about points of failure of medical machine learning, I think first about things like dataset shift (cite Marzyeh), unknowingly fitting confounders (cite Luke) and healthcare dynamics (cite Griffin) instead of true signal, bias (cite Irene), and whether we're preparing for an onslaught of potential overdiagnosis and potentially job displacement.  Given recent issues generalizing (cite recent Topol) to new populations, there are also uncomfortable questions to ask about how confident what type of evidence we should trust before deploying ML in new patient populations.

While all of these issues have general policy concerns, the way I think about them most is in context of how they inform our evaluations of individual ML systems. Each of the above issues demands that specific questions be asked of the systems that we're evaluating.  Questions like:  what population was this model fit on, and how does it compare to the population the system will be used in?  How could the data I'm feeding this algorithm have changed in the time since the model was developed?  Have we thought carefully about the workflow so these algorithms are getting applied to patients with the right priors and the healthcare providers know how to properly act upon positive tests when the time comes?

Our goal in this work was in many ways simply to point out that adversarial attacks at least deserve ackowledgement as one of these potential pitfalls. Questions this reality might prompt us to ask when evaluating a specific system include:  Is there a mismatch in incentives between the person developing/hosting the algorithm and the person sending data into that algorithm?  If so, are we prepared for the fact that those providing data to the algorithm might try to intentionally craft that data to achieve the results they want?  If we decide to try to use models more robust to adversarial attacks, to what extent are we comfortable trading off accuracy in order to do so?

In many application settings, the answer to the incentives question may simply be "no."  But I don't think that's necessarily the case for all possible applications of machine learning in healthcare.  To boot, we as authors have been slightly disconcerted by the fact that when speaking to high-level decision makers at hospitals, insurance companies, and elsewhere who are investing heavily in ML, they generally aren't even aware of the existance of adversarial examples.  So it's really that mismatch in awareness relative to other pitfalls of ML that prompted the paper, even if in the grand scheme of things adversarial attacks are just one piece of a very large pie.



### There seems to have been something of a pivot between the preprint and the policy forum discussion, with the latter focusing much less on images.  Was this intentional?

Yes!  Our preprint was geared toward a technical audience, and was largely motivated by a desire to get people who work on ML security/robustness research to start thinking about healthcare when considering attacks and defenses, rather than just things more native to the CS world like self-driving cars.  At the time, the bulk of high-profile work -- both in adversarial attacks and in medical ML -- had been done in the computer vision space, so we decided to focus on this for our initial deep dive and in building our three proofs of concept.

As we thought a lot more deeply about the problem, however, we realized that we should probably expand our scope.  The bulk of ML happening *today* in the healthcare industry isn't in the form of diagnostics algorithms, but is being used internally at insurance companies to process claims directly for first-pass approvals/denials. And the best examples for existing adversarial attack-like behavior takes place in context of providers manipulating these claims. These provide a jumping off point to understand a spectrum of emerging motivations for adversarial behavior across all aspects of the healthcare system and across many different forms of ML. (See the next section on this as well.)



### In the paper, you frame existing examples like upcoding and claims craftsmanship as adversarial attacks, or at least their precursors.  Is that fair?

I think so. The paper "adversarial classification" from KDD '04 even talks specifically about fraud detection along with spam and other applications of adversarial attacks.

For a few years, the adversarial examples community focused really heavily on *human-imperceptible* changes to *images,* usually computed using gradient tricks.  But more recently, I think the community has (appropriately) returned to something closer to what was described in the original papers, namely any method employed to craft one's data to influence the behavior of an ML algorithm that processes it.  As [Gilmer et al](https://arxiv.org/pdf/1807.06732.pdf) say, "what is important about an adversarial example is that an adversary supplied it, not that the example itself is somehow special."  Such framings of the problem allow even for normal data identified through simple techniques like guess-and-check and grid search to be adversarial examples so long as they are used with adversarial intent, and indeed some recent papers in major CS venues have employed such techinques.

At present, the adversarial behavior in context of things like medical claims appears to be limited to providers stumbling upon or essentially guess-and-checking combinations of codes that will provide higher rates of reimbursement/approval without commiting overt fraud.  (Some [studies like this one](https://jamanetwork.com/journals/jama/fullarticle/192577) have suggested a hefty cohort of physicians think that manipulating claims is even *necessary* in order to provide high-quality care.) In light of the last paragraph, I think you can make a reasonable case that this behavior itselft already constitues an adversarial attack on the ML systems used by insurance companies, though admittedly a fairly boring one from a technical point of view. But it may be getting more interesting. Hospitals invest *immense* resources in this process -- up to [$99k per physician per year](https://jamanetwork.com/journals/jama/article-abstract/2673148?redirect=true) -- and I know for a fact that some providers are already investing heavily in software solutions to more explicitly optimize this stuff.  Likewise, [insurance companies](https://www.forbes.com/sites/insights-intelai/2019/02/11/how-ai-can-battle-a-beastmedical-insurance-fraud/#20fa437e363e) are doubling down on AI solutions to fraud detection, including processing not just claims but things like medical notes. Now that computer vision algorithms are starting to get [FDA approved](https://www.fda.gov/newsevents/newsroom/pressannouncements/ucm604357.htm) for medical purposes, I think it's also likely that payors and regulators will start leveraging this tech as well, which may lead to incentives for computer vision adversarial attacks, a hypothetical scenario at the center of our preprint.

In any event, the real motivation for the claims examples we focus on in the paper is not to call these out as adversarial attacks per se.  Rather, it's to demosntrate how motivations -- both positive and negative -- already exist in the healthcare system that motivate various players to subtly manipulate their data in order to achieve specific results. This is the soul of the adversarial attacks problem.  As both the reach and sophistication of medical machine learning expands across the healthcare system, the techniques used to game these algorithms will likely expand significantly as well.



### "Adversarial attacks" sounds scary.  Do you think people will use these as tools to hurt people by hacking diagnostics, etc?

While this is may be possible in certain circumstances in theory, I don't think it's particularly likely.  By analogy, [pacemaker hacks](https://www.wired.com/story/pacemaker-hack-malware-black-hat/) have been around for more than a decade, but I 



### Are you trying to stall the development of medical ML?

Nope!  Every author on this paper is very bullish on machine learning as a way to achieve positive impact in all aspects of the healthcare system.  We explicitly state this in the paper, as well as the fact that we don't think these concerns should slow things down, just be a part of an ongoing conversation.
