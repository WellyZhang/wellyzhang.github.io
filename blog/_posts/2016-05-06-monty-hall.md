---
layout:     post
title:      "The Monty Hall Problem: A Bayesian Approach"
subtitle:   "How I Attacked a Probabilistic Puzzle"
date:       2016-05-06
author:     "Chi Zhang"
header-img: "img/banner/post-banner-mthall.jpg" 
catalog: true
use-math: true
tags:
    - iNotes  
    - Stats
---

## How I Came Across Monty Hall

It is actually pretty straightforward: a similar problem showed up in Problem Set 1 of the course *Introduction to Data Mining*. It all came as follows:

> There are three boxes. Only one box contains a special prize that will grant you 1 bonus points. After you have chosen a box B1 (B1 is kept closed), one of the two remaining boxes will be opened (called B2) such that it **must not** contain the prize (note that there is at least one such box). Now you are are given a second chance to choose boxes. You can either stick to B1 or choose the only left box B3. What is your best choice?

Well, for those of you familiar with ***Monty Hall Problem***, this actually **IS** one with only modifications of the background.

FYI, the Monty Hall Problem<sup>[[1]](#ref1)</sup> is first proposed in the *Parade* magazine.

> Suppose you're on a game show, and you're given the choice of three doors: Behind one door is a car; behind the others, goats. You pick a door, say No. 1, and the host, who knows what's behind the doors, opens another door, say No. 2, which has a goat. He then says to you, "Do you want to pick door No. 3?" Is it to your advantage to switch your choice?

So, anyway, what is your choice?

## Bayesian Statistics

Allright, before we proceed to the solution, let's first have a brief review of Bayesian Statistics. 

The following equation, or [**Bayes' Theorem**](https://en.wikipedia.org/wiki/Bayes%27_theorem), lies at the heart of Bayesian Statistics,

$$P(A \mid B) = \frac{P(B \mid A) \times P(A)}{P(B)}.$$

It basically is just a reformulation of the condition probability equation, 

$$P(A, B) = P(A \mid B) \times P(B) = P(B \mid A) \times P(A).$$

Under independency, the equation just reduces to

$$P(A, B) = P(A) \times P(B),$$

or equivalently,

$$P(A \mid B) = P(A) \quad P(B \mid A) = P(B).$$

As for $P(B)$,

$$P(B) = \int P(b) {\rm d}b.$$

OK, now that you are equipped with fundamentals of statistics, let's move on!

## The Bayesian Approach

Let's view the Monty Hall problem in its original scenario.

To decide whether we should switch, we should compare probabilities of $P(No. 1 = car \mid No. 2 = goat)$ and $P(No. 3 = car \mid No. 2 = goat)$. And these are just conditional probabilites that require the Bayes' Theorem. Note that

$$P(No. 3 = car \mid No. 2 = goat) = 1 - P(No. 1 = car \mid No. 2 = goat).$$

Thus, we only need to work out $P(No. 1 = car \mid No. 2 = goat)$.

Using the Bayes's Theorem,

$$P(No. 1 = car \mid No. 2 = goat) = \frac{P(No. 2 = goat \mid No. 1 = car) \times P(No. 1 = car)} {P(No. 2 = goat)}.$$


The probability of No. 1 containing the car is

$$P(No. 1 = car) = {1 \over 3},$$

while the probability of the door the host opens containing the goat given that your choice contains the car could be easily derived as

$$P(No. 2 = goat \mid No. 1 = car) = 1.$$

Well, until now, nothing fancy. But here comes the trickiest part.

What is $P(No. 2 = goat)$?

One who thinks little of it might carelessly concludes $2 \over 3$ since the probability of every door containing the car is $1 \over 3$ and that of a goat is just the complement.

**Excuse me, but that's not true.**

Notice in the statement that the host always opens a door that **does not contain a car**.

Therefore the probability is actually $1$.

As a result,

$$P(No. 1 = car \mid No. 2 = goat) = \frac{1 \times {1 \over 3}}{1} = {1 \over 3}.$$

AH! It seems that **we should switch**!

Quite counterintuitive!

## The Frequentist Approach

It was actually a bit confusing when I made my first attempt to understand it from a Bayesian view, although it turned out easier if you simply listed all cases and counted the number of wins.

Let me show you.

| Behind No. 1 | Behind No. 2 | Behind No. 3 | Result if stick | Result if switch |
| :---: | :---: | :---: | :---: | :---: |
| Car | Goat | Goat | **Win** | Lose |
| Goat | Car | Goat | Lose | **Win** |
| Goat | Goat | Car | Lose | **Win** |

<small class="img-hint">Table from [[1]](#ref1)</small>

As illustrated above, $P(No. 1 = car \mid Door = goat) = {1 \over 3}$.

## Naive Generalization

Now that we have worked out the case when there are only 3 doors with only 1 containing the prize, it's natural that we seek the solution when there are more doors and more than 1 prizes.

Similarly, if there is 1 prize and n doors, we should again switch since the Bayes' Theorem tells that the probability of wins remains $1 \over n$.

But the problem becomes a little bit complicated when there are more than 1 prizes.

Suppose there are m prizes and n doors. Then the probability of winning if you stick turns to be 

$$ P = {m \over n}.$$

If this is larger than $1 \over 2$ then we stick, otherwise we switch.

## References
1. <a id="ref1">[Monty Hall problem - Wikipedia, The Free Encyclopedia](https://en.wikipedia.org/wiki/Monty_Hall_problem)</a>








