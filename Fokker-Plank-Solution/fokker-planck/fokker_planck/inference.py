#!/usr/bin/env python

import numpy as np
import pickle

class inference:
	'''
	This is the base class that contains the shared codebase for both the
	linear and nonlinear integrators
	'''
	def __init__(self,parameters={}):
		'''
		Set simulation parameters, which are provided
		as dictionary in the argument parameters
		'''
		#
		self.trajectories_loaded = False
		self.index_loaded = False
		#
		self.verbose = True # default value for verbose
		self.index_directory = './' # default value for index directory
		#
		# the following parameters need to be set by the user
		self.trajectories_filename = 'not set'
		self.xl = -np.inf
		self.xr = +np.inf
		self.N_bins = 100
		self.dt = 'not set'
		# the following two parameters are calculated by the program
		# and cannot be set manually
		self.dx = 'not set'
		self.x = 'not set'
		#
		self.set_parameters(parameters)

	def set_parameters(self,parameters={}):
		'''
		Change parameters of an existing instance of this class
		'''
		#
		try:
			self.verbose = parameters['verbose']
		except KeyError:
			pass
		#
		try:
			self.trajectories_filename = parameters['trajectories_filename']
		except KeyError:
			pass
		#
		try:
			self.index_directory = parameters['index_directory']
		except KeyError:
			pass
		#
		try:
			self.xl = parameters['xl']
		except KeyError:
			pass
		#
		try:
			self.xr = parameters['xr']
		except KeyError:
			pass
		#
		try:
			self.N_bins = parameters['N_bins']
		except KeyError:
			pass
		#
		try:
			self.dt = parameters['dt']
		except KeyError:
			pass
		#
		# update arrays
		#
		if self.xl > -np.inf and self.xr < np.inf:
			self.dx = (self.xr - self.xl)/self.N_bins
			if self.N_bins != 'not set':
				self.x = self.xl + (np.arange(self.N_bins) + 0.5)*self.dx

	def get_parameters(self,print_parameters=False):
		'''
		Return current parameters
		'''
		self.index_parameters = {
				'trajectories_filename':self.trajectories_filename,
				'index_directory':self.index_directory,
				'xl':self.xl,
		          'xr':self.xr,
		          'dx':self.dx,
		          'dt':self.dt,
		          'N_bins':self.N_bins,
		          'x':self.x
				  }
		if print_parameters:
			print("Parameters set for this instance:")
			print("trajectories_filename = {0}".format(
						self.trajectories_filename))
			print("index_directory       = {0}".format(
					self.index_directory))
			print("xl        = {0}".format(self.xl))
			print("xr        = {0}".format(self.xr))
			print("dx        = {0}".format(self.dx))
			print("dt        = {0}".format(self.dt))
			print("N_bins    = {0}".format(self.N_bins))
		return self.index_parameters

	def save_parameters(self):
		'''
		Save current parameters
		'''
		#
		pickle.dump( self.get_parameters(print_parameters=False),
						open( self.index_directory + "/index_parameters.pkl", "wb" ) )

	def load_parameters(self):
		'''
		Load saved parameters
		'''
		#
		index_parameters = pickle.load(
						open( self.index_directory + "/index_parameters.pkl", "rb" ) )
		self.set_parameters(index_parameters)

	def check_if_all_parameters_are_set(self):
		'''
		Check if all the parameters necessary for inference are set
		'''
		#
		if self.trajectories_filename == 'not set':
			raise RuntimeError("Filename for input trajectories not set.")
		#
		if self.dt == 'not set':
			raise RuntimeError("Timestep not set.")

	def load_trajectories(self):
		#
		self.trajectory_lengths = []
		#
		if self.trajectories_filename == 'not set':
			raise RuntimeError("No filename for trajectories to load "\
						+ "provided.")
		#
		self.trajectories = pickle.load(open(f"./data/{self.trajectories_filename}", "rb"))
		#
		for i, current_trajectory in enumerate(self.trajectories):
			self.trajectory_lengths.append(len(current_trajectory))
		if self.verbose:
			print("Loaded {0} trajectories.".format(len(self.trajectories)))
		#
		self.trajectories_loaded = True
		#
		self.get_range_of_time_series()
		self.check_bounds_of_spatial_domain()

	def import_trajectories(self,trajectories):
		#
		self.trajectories = []
		self.trajectory_lengths = []
		#
		for i, current_trajectory in enumerate(trajectories):
			self.trajectories.append(current_trajectory)
			self.trajectory_lengths.append(len(current_trajectory))
		#
		if self.verbose:
			print("Imported {0} trajectories.".format(len(self.trajectories)))
		#
		self.trajectories_loaded = True
		#
		self.get_range_of_time_series()
		self.check_bounds_of_spatial_domain()

	def get_range_of_time_series(self):
		#
		if self.trajectories_loaded == False:
			raise RuntimeError("Please load trajectories first.")
		#
		max_pos = -np.inf
		min_pos = np.inf
		for i, current_trajectory in enumerate(self.trajectories):
			cur_max = np.max(current_trajectory)
			cur_min = np.min(current_trajectory)
			if cur_max > max_pos:
				max_pos = cur_max
			if cur_min < min_pos:
				min_pos = cur_min
		'''
		if self.verbose:
			print("For currently loaded trajectories, minimal and maximal "\
			+ "positions are {0:3.5f} and {1:3.5f}".format(min_pos,max_pos))
		''';
		self.min_pos = min_pos
		self.max_pos = max_pos

	def check_bounds_of_spatial_domain(self):
		#
		if self.xl < self.min_pos:
			if self.xl != -np.inf:
				print("Warning: l0 < (minimal position in dataset), i.e." \
					+ " {0:3.5f} < {1:3.5f}.".format(self.xl,self.min_pos) \
					+ "\n\tUsing l0 = {0:3.5f}".format(self.min_pos))
			self.xl = self.min_pos
		#
		if self.xr > self.max_pos:
			if self.xr != np.inf:
				print("Warning: r0 > (maximal position in dataset), i.e." \
					+ " {0:3.5f} > {1:3.5f}.".format(self.xr,self.max_pos) \
					+ "\n\tUsing r0 = {0:3.5f}".format(self.max_pos))
			self.xr = self.max_pos
		# update parameters so that dx gets calculated again if
		# self.xl or self.xr have changed
		self.set_parameters()

	def get_histogram(self,N_hist=100):
		#
		bin_edges = np.linspace(self.xl,self.xr,
									endpoint=True,
									num=N_hist+1,
									dtype=float)
		bin_centers = (bin_edges[1:] + bin_edges[:-1])/2.
		dx_bins = bin_edges[1] - bin_edges[0]
		hist = np.zeros(N_hist,dtype=float)
		#
		for i, current_trajectory in enumerate(self.trajectories):
			#
			bin_numbers = (current_trajectory - bin_edges[0]) // dx_bins
			for j in range(N_hist):
				hist[j] += np.sum( bin_numbers == j )
		#
		return hist, bin_edges


	def create_index(self):
		#
		# check that trajectories are loaded
		if self.trajectories_loaded == False:
			raise RuntimeError("Please load trajectories first.")
		#
		#
		# traj_number:    enumerates the trajectories we consider
		# traj_index:     enumerates the timestep within each trajectory
		# traj_bin_index: contains the bin index of the current position of the current trajectory
		#
		self.traj_number = np.array([],dtype=int)
		self.traj_index = np.array([],dtype=int)
		self.traj_bin_index = np.array([],dtype=int)
		#
		update_frequency = np.max([len(self.trajectories) // 100,1])
		#
		for i,current_trajectory in enumerate(self.trajectories):
			if self.verbose:
				if i % update_frequency == 0:
					print('Creating index. '\
							+ 'Processing trajectory {0} of {1}..'.format(
								i+1,len(self.trajectories)
									),end='\r')
			current_indices = np.array((current_trajectory-self.xl)//self.dx,
										dtype=int)
			#
			# number of current trajectory
			self.traj_number = np.append(self.traj_number,
										np.ones(len(current_indices))*i)
			# positions within current trajectory
			self.traj_index = np.append(self.traj_index,
										np.arange(len(current_indices)))
			# index of bins
			self.traj_bin_index = np.append(self.traj_bin_index,
										current_indices)
			#
		self.traj_bin_indices = []
		for i in range(self.N_bins):
			self.traj_bin_indices.append(
						np.array(np.where(self.traj_bin_index == i)[0],dtype=int)
										)
		#
		self.traj_number = self.traj_number.astype(int)
		self.traj_index = self.traj_index.astype(int)
		#
		self.index_loaded = True
		#
		if self.verbose:
			print('Finished creating index. Processed {0} trajectories. '.format(
						len(self.trajectories)
							),end='\n')
		#
		#return traj_number ,traj_index, traj_bin_indices

	def save_index(self):
		#
		if self.index_loaded == False:
			raise RuntimeError("No index loaded, so no index can be saved.")
		#
		self.save_parameters()
		#
		pickle.dump( self.traj_number,
						open( self.index_directory + "/index_traj_number.pkl", "wb" ) )
		pickle.dump( self.traj_index,
						open( self.index_directory + "/index.pkl", "wb" ) )
		pickle.dump( self.traj_bin_indices,
						open( self.index_directory + "/index_bin_indices.pkl", "wb" ) )
		#

	def load_index(self):
		#
		self.load_parameters()
		#
		self.traj_number = pickle.load(
						open( self.index_directory + "/index_traj_number.pkl", "rb" ) )
		self.traj_index = pickle.load(
						open( self.index_directory + "/index.pkl", "rb" ) )
		self.traj_bin_indices = pickle.load(
						open( self.index_directory + "/index_bin_indices.pkl", "rb" ) )
		#

	def run_inference(self,N_shift=1):
		#
		D_array = np.zeros(self.N_bins,dtype=float)
		a_array = np.zeros(self.N_bins,dtype=float)
		#
		for i in range(self.N_bins):
			if self.verbose:
				print('Running inference. Processing bin {0} of {1}..'.format(
								i+1,self.N_bins),end='\r')
			D_array[i], a_array[i] = self.kramers_moyal_single_bin(
									bin_index=i,
									N_shift=N_shift)
		if self.verbose:
			print('Finished inference with {0} bins.                   '.format(
											self.N_bins),end='\n')
		#
		output_dictionary = {'x':self.x,
							'D':D_array,
							'a':a_array}
		return output_dictionary

	def kramers_moyal_single_bin(self,
								bin_index,N_shift):
		# get list of trajectories starting in given bin
		list_of_trajectories_starting_in_bin = self.traj_bin_indices[bin_index]
		N = len(list_of_trajectories_starting_in_bin)
		# set up variables for <x>, <x^2>, which we want to estimate from data
		delta_x = 0.
		delta_x_squared = 0.
		# iterate through all trajectories that start in bin
		for i,cur_index in enumerate(list_of_trajectories_starting_in_bin):
			# note that traj_index[cur_index] labels the time at which the trajectory
			# with number traj_number[cur_index] is in the bin of interest.
			#
			if (self.traj_index[cur_index]+N_shift) >= \
				self.trajectory_lengths[self.traj_number[cur_index]]:
				# this means that the length of the current trajectory segment
				# is less than the lagtime we want to use. In that case we
				# skip the current trajectory segment, but since we now have
				# one datapoint less we need to subtract 1 from N
				N -= 1
				continue
			#
			cur_diff = self.trajectories[self.traj_number[cur_index]]   \
								[self.traj_index[cur_index]+N_shift]   \
					- self.trajectories[self.traj_number[cur_index]]   \
										[self.traj_index[cur_index]]
			#
			delta_x += cur_diff
			delta_x_squared += cur_diff**2
		#
		if N == 0:
			print('Warning: Encountered a bin with N = 0 datapoints. To avoid '\
				+ 'this issue, please i) change the interval size [xl,xr], '\
				+ 'ii) decrease N_shift, or iii) use longer '\
				+ 'trajectories.')
			return np.nan, np.nan
		#
		D = delta_x_squared /(2*N*self.dt*N_shift)
		drift = delta_x / (N*self.dt*N_shift)
		#
		return D, drift
