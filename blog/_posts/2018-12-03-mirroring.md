---
layout:     post
title:      "Mirroring without Overimitation"
date:       2018-12-03
author:     "Chi Zhang"
header-img: "img/banner/post-banner-mirroring.jpg"
catalog: true
use-math: true
tags: 
    - Reinforcement Learning
---

> To learn more about the approach we take to teach a robot to open medicine bottles, please checkout our [paper](./attach/aaai19liu.pdf).

## 1. Introduction

A hallmark of machine intelligence is the capability to adapt to new tasks rapidly and "achieve goals in a wide range of environments". In comparison, a human can quickly learn new skills by observing other individuals, expanding their repertoire swiftly to adapt to the ever-changing environment. To emulate the similar learning process, the robotics community has been developing the framework of ***Learning from Demonstration***, *i.e.*, LfD. However, the "correspondence problem", *i.e.*, the difference of embodiments between a human and a robot, is rarely addressed in the prior work of LfD. As a result, a one-to-one mapping is usually handcrafted between the human demonstration and the robot execution, restricting the LfD only to mimic the demonstrator's low-level motor controls and replicate the (almost) identical procedure to achieve the goal. Such behavior is analogous to a phenomenon called "overimitation" observed in human children. 

Obviously, the skills learned from overimitation can hardly be adapted to new robots or new situations.

Inspired by the idea of mirror neurons, we propose a ***mirroring*** approach that extends the current LfD, through the physics-based simulation, to address the correspondence problem. Rather than overimitating the motion controls from the demonstration, it is advantageous for the robot to seek ***functionally equivalent*** but possibly visually different actions that can produce the same effect and achieve the same goal as those in the demonstration.

To achieve this goal, we take a **force-based** approach and deploy a low-cost tactile glove to collect human demonstration with fine-grained manipulation forces. Beyond visually observable space, these tactile-enabled demonstrations capture a deeper understanding of the physical world that a robot interacts with, providing an extra dimension to address the correspondence problem. This approach is also **goal-oriented** in the sense that a "goal" is defined as the desired state of the target object and encoded in a grammar model. We learn a grammar model from demonstrations and allow the robot to reason about the action to achieve the goal state based on the learned grammar.

To show the advantage of our approach, we mirror the human manipulation actions of opening medicine bottles with a child-safety lock to a real Baxter robot. The challenge in this task lies in the fact that opening such bottles requires to push or squeeze various parts, which is visually similar to opening one without a child-safe lock.

## 2. Pipeline

The following figure shows the pipeline of the mirroring approach.

![pipeline](/img/in-post/mirroring/system_diagram.png)
<small class="img-hint">Figure 1. A robot mirrors human demonstrations with functional equivalence by inferring the action that produces similar force, resulting in similar changes of the physical states. Q-Learning is applied to associate types of forces with the categories of the object state changes to produce human-object-interaction (hoi) units.</small>

After collecting human demonstrations on a specific task (here opening bottles with child-safety lock), we first learn by Q-Learning to associate actions (in the space of forces) and object state changes. The learned policy is further transformed into a grammar representation, *i.e.*, T-AoG. During execution, the robot reasons about the action to take by imagining the result of this action and picks one that would lead to a state change towards the final goal.

## 3. Learning Force and State Association

For ease of implementation, we use distributions of forces on objects as the state space of forces and apply K-means clustering to group robot actions. Force distributions in a group are then averaged and normalized. For state representation, we discretize the distance and angle of the bottle lid and normalize them into [0, 1]. Finally, we apply the famous Q learning rule in a temporal difference manner to learn the force and state association.

$$Q(s_i, a_i) = (1 - \alpha) Q(s_i, a_i) + \alpha [r(s_i, a_i) + \gamma \max_k Q(s_{i + 1}, a_k)]$$

## 4. Learning Goal-Oriented Grammar

The human-object-interaction (hoi) sequence learned by policy naturally forms the space of parse sentences from an implicit grammar. Therefore, we could recover the grammar structure by ADIOS following the posterior probability.

$$p(G | X) \propto p(G) p(X | G) = \frac{1}{Z} e^{-\alpha ||G||} \prod_{pt_i \in X} p(pt_i | G)$$

## 5. Mirroring to Robots


