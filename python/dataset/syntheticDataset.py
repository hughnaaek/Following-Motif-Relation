__all__ = ['leading_ts1', 'following_ts1','leading_ts','following_ts']

import numpy as np

def leading_ts1(randseed=0,k_motif=2,varied_len=False,one_motif=False):

  np.random.seed(randseed)

  #-------------------------------------------------------------------------------------------

  if varied_len:

    high = 1000
    N = np.random.randint(int(0.1*high),high=high)

    motif_start_list   = [0]
    discord_start_list = [0]

    while (motif_start_list[-1]<N) and (discord_start_list[-1]<N):
     
      # motif length
      motif_length   = np.random.randint(int(N*0.025),high=int(N*0.1))

      # discord length
      discord_length = int(motif_length*5)
      while (N%(discord_length+motif_length) != 0) and (discord_length>int(N*0.6)): discord_length += 1

      #new motif and discord start
      discord_start_list.append(motif_start_list[-1]+motif_length)
      motif_start_list.append(discord_start_list[-1]+discord_length)

    discord_start_list = discord_start_list[1:]
    if motif_start_list[-1] > N:
      motif_start_list = motif_start_list[:-1]
      if discord_start_list[-1] > N:
        discord_start_list = discord_start_list[:-1]

  else:
    N = 1000

    if one_motif:
      motif_length   = np.random.randint(int(N*0.1),high=int(N*0.15))
      discord_length = int(motif_length*5)
      motif_start_list   = [int(N*np.random.randint(int(15),high=int(30))/100)+motif_length]

    else:
      motif_length   = np.random.randint(int(N*0.025),high=int(N*0.1))
      discord_length = int(motif_length*1.2)
      motif_start_list   = [i for i in range(0,N,int(motif_length+discord_length))]

  t = np.arange(N)
  time_series = np.zeros(N)

  #-------------------------------------------------------------------------------------------

  n = 0
  i = 0
  motif = [[np.random.randint(low=2,high=20)/10,np.random.randint(low=300,high=500)/100] for i in range(k_motif)]
  #motif = [[np.random.randint(low=2,high=10)/10,np.random.randint(low=300,high=500)/10] for i in range(k_motif)]

  if motif_start_list[0] != 0:
    time_series[0:motif_start_list[0]] += (np.random.normal(0, 0.5,len(time_series[0:motif_start_list[0]])))

  while n<len(motif_start_list):
    if k_motif <= 1: i = 0
    try:
      time_series[motif_start_list[n]:motif_start_list[n]+motif_length] += np.sin(motif[i][0] * np.pi * t[:motif_length]/motif[i][1])
    except ValueError:
      time_series[motif_start_list[n]:N] += np.sin(motif[i][0] * np.pi * t[:int(N-motif_start_list[n])]/motif[i][1])
      break

    try:
      time_series[motif_start_list[n]+motif_length:motif_start_list[n+1]] += (np.random.normal(0, 0.5, discord_length))
    except IndexError:
      time_series[motif_start_list[n]+motif_length:N] += (np.random.normal(0, 0.5, (N-motif_start_list[n]-motif_length)))

    n += 1
    i += 1
    if i == len(motif): i = 0

  #-------------------------------------------------------------------------------------------

  time_series += (np.random.normal(0, 0.1, N))
  time_series += (np.random.normal(0, 0.1, N))

  #-------------------------------------------------------------------------------------------
  if not varied_len:
    motif_interval = [[x for x in motif_start_list],[x+motif_length for x in motif_start_list]]
    

    return (time_series, motif_interval)

  return time_series


def following_ts1(leading_ts,randseed=0,uncontinuous=True,ground_truth=False):

  np.random.seed(randseed)

  leading_signal = leading_ts[0].copy()
  N              = (leading_signal.shape)[0]
  percent        = np.random.randint(low=20,high=35)/100
  N_first_signal = int(np.ceil(N*percent))
  N_follow       = N - N_first_signal

  degree_of_noise = 3
  noise = 0.5

  first_signal = np.random.normal(0, noise, N_first_signal)

  #leader interval
  lead_motif_start = list(leading_ts[1][0])
  lead_motif_stop  = list(leading_ts[1][1])

  motif_start,motif_stop = [],[]

  #following time series generating
  if uncontinuous:

    time_series = np.zeros(N)
    cutting_point = int(len(time_series)/2)

    percent        = np.random.randint(low=10,high=15)/100
    second_N       = int(len(time_series)*percent)

    interrupt = True

    for t in range(N):
      lag = np.random.randint(0, min(t, degree_of_noise) + 1)
      time_series[t] = leading_signal[t - lag]

      if t in lead_motif_start:
        if t<cutting_point:
          motif_start.append(t+N_first_signal)
        else:
          motif_start.append(t+N_first_signal+second_N)

      if t in lead_motif_stop:
        if t<cutting_point:
          motif_stop.append(t+N_first_signal)
        elif (t in range(cutting_point,cutting_point+second_N+1)) and interrupt:
          motif_stop.append(cutting_point+N_first_signal)
          interrupt = False
        else:
          motif_stop.append(second_N+N_first_signal+t)

    time_series1 = time_series[:cutting_point]
    time_series2 = time_series[cutting_point:]

    percent        = np.random.randint(low=10,high=15)/100
    second_signal = np.random.normal(0, noise, second_N)

    time_series = np.hstack((first_signal,time_series1,second_signal,time_series2))[:N]

  else:
    time_series = np.zeros(N)
    for t in range(N):
      lag = np.random.randint(0, min(t, degree_of_noise) + 1)
      time_series[t] = leading_signal[t - lag]

      if t in lead_motif_start:
        motif_start.append(t+N_first_signal)
      if t in lead_motif_stop:
        motif_stop.append(t+N_first_signal)

    time_series = np.hstack((first_signal,time_series))[:N]

  # adding noise
  time_series += np.random.normal(0, 0.1, N)
  time_series += (np.random.normal(0, 0.1, N))

  #motif_interval
  motif_start = [x for x in motif_start if x<N]
  motif_stop = [x for x in motif_stop if x<N]

  motif_interval = [motif_start[:len(motif_stop)],motif_stop]
  #print(motif_interval)

  if ground_truth:
    motif_start = motif_start[:len(motif_stop)]

    keys   = list(range(len(motif_start)))
    values = []
    for i in range(len(motif_start)):
      pair_result = {
          "leader_interval" : [lead_motif_start[i],lead_motif_stop[i]],
          "follower_interval":[motif_start[i],motif_stop[i]]
      }
      values.append(pair_result)
    result = dict(zip(keys,values))

    return (time_series, result)

  return (time_series, motif_interval)


def leading_ts(randseed=0,k_motif=2,varied_len=False,one_motif=False):

  np.random.seed(randseed)

  #-------------------------------------------------------------------------------------------

  if varied_len:

    high = 1000
    N = np.random.randint(int(0.1*high),high=high)

    motif_start_list   = [0]
    discord_start_list = [0]

    while (motif_start_list[-1]<N) and (discord_start_list[-1]<N):
     
      # motif length
      motif_length   = np.random.randint(int(N*0.025),high=int(N*0.1))

      # discord length
      discord_length = int(motif_length*5)
      while (N%(discord_length+motif_length) != 0) and (discord_length>int(N*0.6)): discord_length += 1

      #new motif and discord start
      discord_start_list.append(motif_start_list[-1]+motif_length)
      motif_start_list.append(discord_start_list[-1]+discord_length)

    discord_start_list = discord_start_list[1:]
    if motif_start_list[-1] > N:
      motif_start_list = motif_start_list[:-1]
      if discord_start_list[-1] > N:
        discord_start_list = discord_start_list[:-1]

  else:
    N = 1000

    if one_motif:
      motif_length   = np.random.randint(int(N*0.1),high=int(N*0.15))
      discord_length = int(motif_length*5)
      motif_start_list   = [int(N*np.random.randint(int(15),high=int(30))/100)+motif_length]

    else:
      motif_length   = np.random.randint(int(N*0.025),high=int(N*0.1))
      discord_length = int(motif_length*1.2)
      motif_start_list   = [i for i in range(0,N,int(motif_length+discord_length))]

  t = np.arange(N)
  time_series = np.zeros(N)

  #-------------------------------------------------------------------------------------------

  n = 0
  i = 0
  motif = [[np.random.randint(low=2,high=20)/10,np.random.randint(low=300,high=500)/100] for i in range(k_motif)]
  #motif = [[np.random.randint(low=2,high=10)/10,np.random.randint(low=30,high=50)] for i in range(k_motif)]

  if motif_start_list[0] != 0:
    time_series[0:motif_start_list[0]] += (np.random.normal(0, 0.5,len(time_series[0:motif_start_list[0]])))

  while n<len(motif_start_list):
    factor = np.random.randint(low=1,high=7)
    if k_motif <= 1: i = 0
    try:
      time_series[motif_start_list[n]:motif_start_list[n]+motif_length] += np.sin(motif[i][0] * np.pi * t[:motif_length]/motif[i][1])
    except ValueError:
      time_series[motif_start_list[n]:N] += np.sin(motif[i][0] * np.pi * t[:int(N-motif_start_list[n])]/motif[i][1])
      break

    #if i%3==0: factor=5
    try:
      time_series[motif_start_list[n]+motif_length:motif_start_list[n+1]] += (np.random.normal(0, 0.5, discord_length))*factor
    except IndexError:
      time_series[motif_start_list[n]+motif_length:N] += (np.random.normal(0, 0.5, (N-motif_start_list[n]-motif_length)))*factor

    n += 1
    i += 1
    if i == len(motif): i = 0

  #-------------------------------------------------------------------------------------------

  time_series += (np.random.normal(0, 0.1, N))
  time_series += (np.random.normal(0, 0.1, N))

  #-------------------------------------------------------------------------------------------
  if not varied_len:
    motif_interval = [[x for x in motif_start_list],[x+motif_length for x in motif_start_list]]
    

    return (time_series, motif_interval)

  return time_series


def following_ts(leading_ts,randseed=0,uncontinuous=True,ground_truth=False):

  np.random.seed(randseed)

  leading_signal = leading_ts[0].copy()
  N              = (leading_signal.shape)[0]
  percent        = np.random.randint(low=20,high=35)/100
  N_first_signal = int(np.ceil(N*percent))
  N_follow       = N - N_first_signal

  degree_of_noise = 3
  noise = 0.5

  first_signal = np.random.normal(0, noise, N_first_signal)*5

  #leader interval
  lead_motif_start = list(leading_ts[1][0])
  lead_motif_stop  = list(leading_ts[1][1])

  motif_start,motif_stop = [],[]

  #following time series generating
  if uncontinuous:

    time_series = np.zeros(N)
    cutting_point = int(len(time_series)/2)

    percent        = np.random.randint(low=10,high=15)/100
    second_N       = int(len(time_series)*percent)

    interrupt = True

    for t in range(N):
      lag = np.random.randint(0, min(t, degree_of_noise) + 1)
      time_series[t] = leading_signal[t - lag]

      if t in lead_motif_start:
        if t<cutting_point:
          motif_start.append(t+N_first_signal)
        else:
          motif_start.append(t+N_first_signal+second_N)

      if t in lead_motif_stop:
        if t<cutting_point:
          motif_stop.append(t+N_first_signal)
        elif (t in range(cutting_point,cutting_point+second_N+1)) and interrupt:
          motif_stop.append(cutting_point+N_first_signal)
          interrupt = False
        else:
          motif_stop.append(second_N+N_first_signal+t)

    time_series1 = time_series[:cutting_point]
    time_series2 = time_series[cutting_point:]

    percent        = np.random.randint(low=10,high=15)/100
    second_signal = np.random.normal(0, noise, second_N)*5

    time_series = np.hstack((first_signal,time_series1,second_signal,time_series2))[:N]

  else:
    time_series = np.zeros(N)
    for t in range(N):
      lag = np.random.randint(0, min(t, degree_of_noise) + 1)
      time_series[t] = leading_signal[t - lag]

      if t in lead_motif_start:
        motif_start.append(t+N_first_signal)
      if t in lead_motif_stop:
        motif_stop.append(t+N_first_signal)

    time_series = np.hstack((first_signal,time_series))[:N]

  # adding noise
  time_series += np.random.normal(0, 0.1, N)
  time_series += (np.random.normal(0, 0.1, N))

  #motif_interval
  motif_start = [x for x in motif_start if x<N]
  motif_stop = [x for x in motif_stop if x<N]

  motif_interval = [motif_start[:len(motif_stop)],motif_stop]
  #print(motif_interval)

  if ground_truth:
    motif_start = motif_start[:len(motif_stop)]

    keys   = list(range(len(motif_start)))
    values = []
    for i in range(len(motif_start)):
      pair_result = {
          "leader_interval" : [lead_motif_start[i],lead_motif_stop[i]],
          "follower_interval":[motif_start[i],motif_stop[i]]
      }
      values.append(pair_result)
    result = dict(zip(keys,values))

    return (time_series, result)

  return (time_series, motif_interval)