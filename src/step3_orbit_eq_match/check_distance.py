# -*- coding: utf-8 -*-
# check_distance.py
###################################################################################################
###################################################################################################
# Python script to search the orbits which meet distance requirment.
# The distance between orbit and epicenter is caculated with Hubeny formula.
# =================================================================================================
# Explanation of each object:
# -------------------------------------------------------------------------------------------------
#[1.    Importing Modules]
# 1-1.  Importing pandas, pathlib, and tqdm.auto
#
#
#[2.  Setting the directory path]
#2-1.  Setting the input and output directory paht.
#2-2.  Show the directory in the terminal.
#
# [3.   Setting the parameters]
# 3-1.  Setting the long radius of the Earth.
# 3-2.  Setting the short radius of the Earth.
# 3-3.  Caculating the eccentricity of the Earth.
#
#
# [4.   Importing the earthquake data from a csv file which contains orbit data metting time 
#       condition]
#4-1.   Importing the earthquake data from a csv file which contains orbit data metting time 
#       condition.
#4-2.   Caculating the length of the data frame object.
#
#
# [5.   Preparation for searching]
# 5-1.  Making a list which contains earthquake and orbit(meeting distance requirement) data.
#
#
# [6.   Searching orbit which meet distance requirement]
# 6-1.      Setting a loop of earthquake data which contains orbit data meeting time condition.
# 6-2.      Making a list which will contain an orbit meeting distance requirement.
# 6-3.      Extracting orbit num(file name) from df.
# 6-4.      Opening the orbit data with orbit num(file name)
# 6-5.      Importing orbit data from a csv file.
# 6-6.      Caculating distance between the orbit and the epicenter.(Setting a loop of orbit data)
# 6-7.      Converting latitude of the epicenter form degeree into radians notation.
# 6-8.      Converting longitude of the epicenter from degerr into radians notation.
# 6-10.     Converting latitude of the orbit from degree into radians notation.
# 6-11.     Converting longitude of the orbit from degree into radians notation.
# 6-13.     Caculating latitude difference between the epicenter and the orbit.
# 6-14.     Caculating longitude difference between the epicenter and the orbit.
# 6-15.     Normalization the longitude from[0,360] to [-180,+180].
# 6-16.     Caculating the average of latitude of the epicenter and the latitude of the orbit.
# 6-17.     Calculating several parameters which are necessary to caculate the distance.
# 6-18.     Calculating the distance between the epicenter and the orbit in km.
# 6-19.     Cheking whether the distance meeting distance requirement.
# 6-20.     
#**************************************************************************************************
# Structure of this script:
#--------------------------------------------------------------------------------------------------
#
#**************************************************************************************************