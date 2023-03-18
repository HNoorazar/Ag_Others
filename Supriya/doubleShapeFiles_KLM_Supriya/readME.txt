Names of files are descriptive:

- onlyAnnuals_JustIrr_LastSrvCorrect_NASSIn_2016_2017_2018_sentinel
   includes fields that have had annual crops, are irrigated, last survey date is up to date, and surveyed by NASS.
   If you do not want NASS toss them out. Kirti does not care about NASS tho. She also does not care about
   last survey date. So, it seems you can use the following file.

- onlyAnnuals_JustIrr_LastSrvFalse_NASSIn_2016_2017_2018_sentinel




onlyAnnuals_JustIrr_LastSrvCorrect_NASSIn_2016_2017_2018@data %>% 
group_by(county) %>% 
summarise(count = n_distinct(ID))
# 1 Adams         240
# 2 Asotin          2
# 3 Benton        470
# 4 Chelan          1
# 5 Columbia        6
# 6 Douglas         3
# 7 Franklin      903
# 8 Garfield        3
# 9 Grant        1150
# 10 Kittitas       21
# 11 Klickitat      56
# 12 Lincoln         5
# 13 Okanogan        5
# 14 Spokane         3
# 15 Stevens         6
# 16 Walla Walla   347
# 17 Whitman         8
# 18 Yakima        420


onlyAnnuals_JustIrr_LastSrvFalse_NASSIn_2016_2017_2018@data %>% 
group_by(county) %>% 
summarise(count = n_distinct(ID))

#   county      count
#    <chr>       <int>
#  1 Adams         536
#  2 Asotin          3
#  3 Benton        827
#  4 Chelan          1
#  5 Columbia       19
#  6 Douglas         4
#  7 Ferry           1
#  8 Franklin     1483
#  9 Garfield       11
# 10 Grant        2234
# 11 Kittitas       29
# 12 Klickitat      82
# 13 Lincoln        38
# 14 Okanogan       20
# 15 Spokane        11
# 16 Stevens         8
# 17 Walla Walla   468
# 18 Whitman        12
# 19 Yakima        995




