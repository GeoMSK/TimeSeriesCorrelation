import pstats
import sys

__author__ = 'gm'

profile_data_file = sys.argv[1]
sort_by = "cumulative"


ps = pstats.Stats(profile_data_file).sort_stats(sort_by)
ps.print_stats()
