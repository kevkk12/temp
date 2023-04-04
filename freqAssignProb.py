import pandas as pd
import numpy as np
import math
from ortools.sat.python import cp_model


def getCoChannelMatrix(fileName):

  df = pd.read_excel(fileName)
  # Get the station numbers
  stations = df['N_Station'].unique()

  # Construct zero-based index
  stationsIndexed = stations - 1

  # Cochannel inteference matrix
  # (i,j) = Percentage inteference on station i due to station j
  coChannelM = np.zeros((len(stations), len(stations)), dtype=int)

  for index, row in df.iterrows():
    # Construct zero-based index
    i = row['N_Station'] - 1 
    j = row['N_Inteferer'] - 1
    # Populate coChannel inteference matrix
    coChannelM[i,j] = math.ceil(row['Pc interf.'])
  
  return coChannelM

def freqAssign(coChannelM):
  numSites = coChannelM.shape[0]
  lowBandStart = 1
  lowBandEnd = 20 #11
  highBandStart = 148
  highBandEnd = 158

  channelListLo = [i for i in range(lowBandStart,lowBandEnd+1)]
  channelListHi = [i for i in range(highBandStart,highBandEnd+1)]

  channelList = channelListLo + channelListHi
  numChannels = len(channelList)
  
  # Model
  model = cp_model.CpModel()

  ## Variables

  # v are sites
  v = []
  # Construct non-contiguous domain for the site frequencies
  vDomain = cp_model.Domain.FromIntervals([[lowBandStart, lowBandEnd], [highBandStart, highBandEnd]])

  for i in range(numSites):
    v.append(model.NewIntVarFromDomain(vDomain, f'v[{i}]'))

  
  ## Constraints

  # Force sites with cochannel intef > threshold to have different frequency assignment.
  # Point of this is to use the AddAllDifferent constraint which should speed up the search process. 
  # TODO: Need to fix this part to make the constraint that x1 != intefering sites. Right now it's saying x1 != x2 != x3 etc...Possibly use forbiddenAssignment?
  # coChannelInterferenceThreshold = 10
  # interferingSitesCumulative = []
  # for i in range(coChannelM.shape[0]):
  #   interferingSites = []
  #   interferingSites.append(v[i])
  #   for j in range(coChannelM.shape[1]):
  #     if coChannelM[i,j] >= coChannelInterferenceThreshold:
  #       interferingSites.append(v[j])
  #   if len(interferingSites)-1 != 0:
  #     model.AddAllDifferent(interferingSites)
  # z = 0

  # Ensure that each site total inteference is less than threshold. i.e. add up all the relevant elements in the row
  # We setup new boolean variables to indicate sites that share the same frequency

  #Channelling https://developers.google.com/optimization/cp/channeling?hl=en
  # vEqual[i,j] = (v[i]==v[j])

  vEqual = [[0] * len(v) for _ in range(len(v))]
  for i in range(coChannelM.shape[0]):
    for j in range(coChannelM.shape[1]):
      vEqual[i][j] = model.NewBoolVar(f'vEqual[{i}][{j}]')

      model.Add(v[i] == v[j]).OnlyEnforceIf(vEqual[i][j])
      model.Add(v[i] == v[j]).OnlyEnforceIf(vEqual[i][j].Not())
  
  for i in range(len(v)):
    model.Add(cp_model.LinearExpr.WeightedSum(vEqual[i], coChannelM[i]) <= 20)

  # Objective
  #objective.append(cp_model.LinearExpr.WeightedSum)
  objective_terms = []

  # Solver
  solver = cp_model.CpSolver()
  status = solver.Solve(model)

  # cp_model.OPTIMAL = 4
  # cp_model.FEASIBLE = 2
  # cp_model.INFEASIBLE = 3
  print(f"Solver status is {status}")

  # Print Solution
  channelListDict = {channel: [] for channel in channelList}
  if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    for site in range(numSites):
      assignedChannel = solver.Value(v[site])
      print(f"Site {site} is assigned channel {assignedChannel}")
      channelListDict[assignedChannel].append(site)

    for channel in channelList:
      print(f"Channel {channel} is used by {channelListDict[channel]}")
    print(f"Total number of channels is {numChannels}")
  else:
      print('No solution found.')

  a = 0



def main():
  fileName = 'interfCalc.xlsx'
  coChannelM = getCoChannelMatrix(fileName)
  freqAssign(coChannelM)


if __name__ == '__main__':
  main()