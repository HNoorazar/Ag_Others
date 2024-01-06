import numpy as np
import pandas as pd

from datetime import date, datetime
from random import seed, random

import os, os.path, shutil, sys, time

start_time = time.time()

####################################################################################
###
###                      Kamiak Core path
###
####################################################################################

####################################################################################
###
###      Parameters
###
####################################################################################

arg1 = int(sys.argv[1])
print(f"Passed Args. are: {arg1=:}!")

####################################################################################
data_base = "/data/project/agaid/h.noorazar/parallel_jobs/"
os.makedirs(data_base, exist_ok=True)

####################################################################################
print("--------------------------------------------------------------")
print(date.today(), "-", datetime.now().strftime("%H:%M:%S"))
error_file_directory = "/home/h.noorazar/educational/error/arg1.o"

df = pd.DataFrame(index=range(5))
df["arg1"] = arg1
df["exponents"] = 0

for counter in df.index:
    if counter == 0:
        print(
            f"This line will be printed in the file {error_file_directory} which is defined in template.sh!"
        )
        print(f"{counter = :}")
    df.loc[counter, "exponents"] = arg1**counter


out_name = data_base + "df_" + str(arg1) + ".csv"
df.to_csv(out_name, index=False)

end_time = time.time()

print("current time is {}".format(time.time()))
print("it took {:.0f} minutes to run this code.".format((end_time - start_time) / 60))


print("--------------------------------------------------------------")
print(date.today(), "-", datetime.now().strftime("%H:%M:%S"))
print("--------------------------------------------------------------")
