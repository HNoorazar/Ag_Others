
The files in here demonstrate how to run jobs in parallel on Kamiak.

If your tasks are independent of each other you can submit one job per task.


1. Edit the file "template.sh". Put your own directories in there.
It is passing an argument to the driver ("parallel_job.py"). If you have more arguments, you need to make adjustments.

2. Edit and run create_all_SHs.sh on Kamiak. Edit so that it includes your directories and correct arguments, etc.

This file will copy "template.sh" to the "/home/..../qsubs" directory. 
One copy per job; if you have 10 jobs, 10 copies will be created.

3. Edit and run "run_all_SHs.sh"
This file will run and submit each of the copies created in step 2, as a new single job.

