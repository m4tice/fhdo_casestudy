from fhdo_casestudy.cs_utils import tools as to
import os
import matplotlib.pyplot as plt

map_csv = "database/town02/town02-gps-intersections.csv"
_ = to.map_plot(map_csv)

plt.show()