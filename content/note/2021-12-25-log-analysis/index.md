---
title: Log Analysis
author: phil e
date: '2021-12-25'
slug: log-analysis
categories: []
tags: []
---
## Project Note

### Problem
Log files are generated everytime our system is turned on.  The issues usually arise when a customer returns from a flight and gives very vague description of a problem.  

What we would like to know is:
* What was planned?
* What worked?  What is NOT working?
* Why is it not working?
* Some standard information from the flight.

### Solutions
What is the output that gives us this information?

End result was using powershell to read and parse the files needed and pull out three (3) distinct sections:

#### Identification / Version Information
* Date
* Mission
* Associated SW Versions
* Sensor ID

#### Test Status

* Find SBIT and Status of all tests with Time Stamp
* TODO: Any failures should indicate Test ID  and information from the SBIT Test Documentation.

### Mission Status
* Calibration Tasks
* Imaging Tasks
* Failed Imaging tasks
* End of mission time


All of this was done using Powershell.
Main factors were:
1. Who will use the tool?  
2. Any licenses needed ?  
3. Any special equipment - laptop needed?
4. Can it work overseas and without internet connection.


Powershell file posted here:  [todo:link here]!





