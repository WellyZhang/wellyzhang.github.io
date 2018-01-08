---
layout:     post
title:      "Pros and Cons of Parsing-Based App"
date:       2016-10-23
author:     "Chi Zhang"
header-img: "img/banner/post-banner-browser.jpg"
catalog: true
tags: 
    - App Development
---

> This a summary of my personal experience on developing and using web-parsing-based tools.

## 1 Introduction

A trend that this new decade has witnessed is that nearly every mature application has its own counterpart on the web and that web users tend not to install a local piece of software but rather rely heavily on the web-based alternatives to access what they want: this, consequently, greatly reduces their burden on maintaining large local disks and at the same time provides a free and rich variety of services users could order just through the browsers. As a result, *WebApp* has now become a new heatedly discussed topic over programmers, start-up founders and entrepreneurs.

No wonder Google introduced [*ChromeBook*](https://www.google.com/chromebook/) years ago and is heavily recruiting front-end and back-end software engineers.

Whether you have realized it or not, the *Internet* has become the biggest resource pool ever that is free and easily accessible.

But the question is: ***how can we make the most of it?***

## 2 Browsers VS. Programs

Most, if not all, non-technicians would clearly answer without a second thought that web browser is a perfect choice to access the Internet. Sure enough, I wouldn't deny it so long as there is no such a thing called program. But let's talk about it latter and first take a look at the pros and cons of the browsers.

### 2.1 Pros and Cons of Browsers

Yes, browsers are interactive, visually intuitive, easily understandable and good for graphs, diagrams and animation; simple clicks on buttons would lead you directly to beautiful pictures and fancy 3D animation.

But there are also problems with the web and browsers. Just to name a few, ads are the biggest problem: they are just everywhere -- either take up much of the space of your screen, suddenly pop up to disturb you from reading or consume much of your Internet speed capacity. There was a time when Chrome was my favorite web browser, but it just held much of my limited memory -- you might know every new tab runs as an independent process. Microsoft IE and Edge are memory-efficient and light-weight, but they are just too slow once you learn to enjoy the superfast Chrome kernel. So time to time, I have to trade off between the memory consumption and the speed.

### 2.2 Python to the Rescue

However, as a programmer, there is always another choice: **programs**.

Though not as fast as C/C++ at high performance scientific computing, Python is already well-known for its pseudocode-like interface, elegant coding style, active support community, rich variety of libraries and most importantly, I/O efficiency, that altogether make it a perfect choice for web-based programming.

As long as you know how, programing would save you from the annoying advertisements, huge consumption of memory and painful speed limit incurred by kernels and sometimes even extend your accessibility to web resources.

Now I'm going to talk about them in greater details.

## 3 Experience Gained from [TermDict](https://github.com/WellyZhang/termdict) and [You-Get](https://github.com/WellyZhang/you-get)

Tired of the disgusting ads when looking up words in Merriam-Webster Dictionary, I decided to write my own web-parsing-based dictionary on console -- TermDict. It took no more than several days of my leisure time: the standard library of ```request``` and the third-party ```BeautifulSoup``` famous for web-parsing and Dom analysis greatly enhanced my coding experiments; the difficulty happened only at understanding the HTML structure -- which information was stored where. But anyway, I came over and the code is open-sourced in my repo. And from then on, I relied heavily on TermDict to look up words: no terrible ads to distract me from reading, no huge memory consumption from the browsers and fast speed as well -- great experience improvement. 

Days later, I found on YouTube an amusing video that I just wanted to have a local copy, but YouTube declares that video downloading is not allowed. I'm not advocating what I did, but technically speaking, there is no way I couldn't get it now that it runs smoothly on my browser. So why not write a highly customized YouTube downloader? I spent several hours on writing code to enable downloading through the web and the only magic worked as the program pretended to be a browser sending a POST message to the server and buffer-by-buffer read what was returned by the HTTP protocol. I once decided to release the code but just encountered You-Get, another web-parsing-based downloader that has a wider range of support for numerous websites. After delving deep into its code, I found we shared similar program logics.

### 3.1 The Good and The Bad

After coding and reading, I do love the idea of web-parsing-based apps enabled by Python, but they also pose serious threats to the problems at hand.

#### 3.1.1 The Bad

##### 3.1.1.1 Difficulty of Parsing

Understanding the DOM structure is the most difficult. Since nowadays web are barely static anymore, meaning that they are generated by back-end programs or databases, parsing by human is just like reverse engineering: you have to figure out which ```<div>``` contains which entry manually and since you are never able to traverse all of the situations, you are nearly sure to introduce bugs into your code. Your code might only work under certain circumstances but not under the other.

##### 3.1.1.2 Maintainability of Code

Now that the structure of the DOM is hard to find, your code is easily populated with ```if```, ```elif``` and ```else``` and without proper comments, it's nearly immediate that your code becomes cluttered and extremely hard for others to understand. Then the problem of maintainability therefore arises.

##### 3.1.1.3 Versatility of the Web

A potential problem with web parsing is that web design changes from time to time and once the structure of the web is changed, part of you code might fail -- it either finds the wrong entry or simple returns a null. This would require frequent, well, at least as fast as how often the web design changes, refactoring of the code and unless you're very enthusiast about the project, fail your code.

#### 3.1.2 The Good

##### 3.1.2.1 Get Rid of API Constraints

APIs are convenient but very limited in another way. APIs are not always free and even if they are, the company exercises constraints on their usage: you have to first become a registered developer; APIs for non-commercial use also come with a limit on the number of times for references; part of the service does not provide APIs. And web-parsing programs just help you get rid of them.

##### 3.1.2.2 Never Get Distracted from Ads

One of the major advantages of web parsing is its concentration on the contents. What you need to do is just extract useful information from the web page, process it and discard all other rubbish, including ads and other indirect links. That just enables an immersive reading environment.

##### 3.1.2.3 Light-Weight and Fast Speed

Web parsing is no more than pieces of code that involves mostly string processing, not much computing and rendering. Thus, the maximum memory requirement would not exceed a few MBs, well, compared to nearly 1GB of Chrome. Besides, the speed of access would also be increased due to the light requirements.

## 4 Conclusion

WebApp might be *the new 42*, offering both accessibility and convenience to its users, and also providing, though sometimes unwillingly, a great opportunities for programmers who want to scrape the web. Using Python, you are more than welcome to use the Internet as the largest ever free resource pool to extract any information you want. Although there are problems relating to the difficulty of parsing, the maintainability of code and the versatility of the web, the API constraints are relaxed with ads gone, light memory consumption and fast speed. The decision is left for you to trade off.

## References

