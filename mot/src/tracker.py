import numpy as np 
from kalmanFilter import KalmanFilter
from scipy.optimize import linear_sum_assignment
from collections import deque

max_trace = 10

class Tracks(object):
	"""docstring for Tracks"""
	def __init__(self, detection, trackId, state):
		super(Tracks, self).__init__()
		self.KF = KalmanFilter()
		self.KF.predict()
		self.KF.correct(np.matrix(detection).reshape(2,1))
		self.trace = deque(maxlen=max_trace)
		self.prediction = detection.reshape(1,2)
		self.trackId = trackId
		self.skipped_frames = 0
		self.trace_state = deque(maxlen=max_trace)
		self.state_correction = state.reshape(1,4)

	def predict(self,detection):
		self.prediction = np.array(self.KF.predict()).reshape(1,2)
		self.KF.correct(np.matrix(detection).reshape(2,1))
	
	def correction(self, state):
		self.state_correction = state.reshape(1,4)


class Tracker(object):
	"""docstring for Tracker"""
	def __init__(self, dist_threshold, max_frame_skipped, max_trace_length):
		super(Tracker, self).__init__()
		self.dist_threshold = dist_threshold
		self.max_frame_skipped = max_frame_skipped
		self.max_trace_length = max_trace_length
		self.trackId = 0
		self.tracks = []

	def update(self, detections, states):
		if len(self.tracks) == 0:
			for i in range(detections.shape[0]):
				track = Tracks(detections[i], self.trackId, states[i])
				self.trackId +=1
				self.tracks.append(track)
		
		N = len(self.tracks)
		M = len(detections)
		cost = []
		for i in range(N):
			diff = np.linalg.norm(self.tracks[i].prediction - detections.reshape(-1,2), axis=1)
			cost.append(diff)

		cost = np.array(cost)*0.1
		row, col = linear_sum_assignment(cost)
		assignment = [-1]*N
		for i in range(len(row)):
			assignment[row[i]] = col[i]
		
		un_assigned_tracks = []

		for i in range(len(assignment)):
			if assignment[i] != -1:
				if (cost[i][assignment[i]] > self.dist_threshold):
					assignment[i] = -1
					un_assigned_tracks.append(i)
				else:
					self.tracks[i].skipped_frames +=1
		
		del_tracks = []
		for i in range(len(self.tracks)):
			if self.tracks[i].skipped_frames > self.max_frame_skipped :
				del_tracks.append(i)
		
		if len(del_tracks) > 0:
			for i in range(len(del_tracks)):
				del self.tracks[i]
				del assignment[i]
		
		for i in range(len(detections)):
			if i not in assignment:
				track = Tracks(detections[i], self.trackId, states[i])
				self.trackId +=1
				self.tracks.append(track)

		for i in range(len(assignment)):
			if(assignment[i] != -1):
				self.tracks[i].skipped_frames = 0
				self.tracks[i].predict(detections[assignment[i]])
				self.tracks[i].correction(states[assignment[i]])
			self.tracks[i].trace.append(self.tracks[i].prediction)
			self.tracks[i].trace_state.append(self.tracks[i].state_correction)


	def printer(self,str):
		print("testline :",str)
		for i in range(len(self.tracks)):
			print(self.tracks[i].trace)
		print()










		



