import pstats

__author__ = 'gm'

profile_data_file = "profiler.data"
sort_by = "cumulative"


ps = pstats.Stats(profile_data_file).sort_stats(sort_by)
ps.print_stats()
