import random

Distribution = (0.2,0.4,0.1,0.1,0.1,0.1)
iteration = 1000
throws=250

def rolling(Distribution):
    randRoll = random.random() 
    sum = 0
    result = 1
    for _ in Distribution:
        sum += _
        if randRoll < sum:
            return result
        result+=1

def dice(Distribution,thrown,iteration):
  numstep = 0
  step=0
  for i in range(iteration):
    for i in range(thrown):
      result = rolling(Distribution)
      if result==1 or result==2:
        step=max(0,step-1)
      elif result>=3 and result<=5:
        step=step+1
      else: 
        result1 = rolling(Distriution)
        step=step+result1
    if step>60: numstep=numstep+1
  return numstep

numstep = dice(Distribution,thrown,iteration)

print("Probability = ",numstep/iteration)
