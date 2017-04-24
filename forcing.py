from __future__ import division
import numpy as np


class Forcing(object):
	"""Forcing of GHG emissions for the EZ-Climate model.

	Attributes
	----------
	sink_start : float
		sinking constant
	forcing_start : float
		forcing start constant
	forcing_p1 : float
		forcing constant
	forcing_p2 : float
		forcing constant 
	forcing_p3 : float
		forcing constant
	absorbtion_p1 : float
		absorbtion constant
	absorbtion_p2 : float 
		absorbtion constant
	lsc_p1 : float
		class constant
	lsc_p2 : float
		class constant 

	"""
	sink_start = 35.596
	forcing_start = 4.926
	forcing_p1 = 0.13173
	forcing_p2 = 0.607773
	forcing_p3 = 315.3785
	absorbtion_p1 = 0.94835
	absorbtion_p2 = 0.741547
	lsc_p1 = 285.6268
	lsc_p2 = 0.88414

	@classmethod
	def _forcing_and_ghg_at_node(cls, m, node, tree, bau, subinterval_len, returning="forcing"):
		"""Calculates the forcing based mitigation or GHG level leading up to the 
		damage calculation in `node`.
		"""
		if node == 0 and returning == "forcing":
			return 0.0
		elif node == 0 and returning== "ghg":
			return bau.ghg_start

		period = tree.get_period(node)
		path = tree.get_path(node, period)

		period_lengths = tree.decision_times[1:period+1] - tree.decision_times[:period]
		increments = period_lengths/subinterval_len

		cum_sink = cls.sink_start
		cum_forcing = cls.forcing_start
		ghg_level = bau.ghg_start

		for p in range(0, period):
			start_emission = (1.0 - m[path[p]]) * bau.emission_by_decisions[p]
			if p < tree.num_periods-1: 
				end_emission = (1.0 - m[path[p]]) * bau.emission_by_decisions[p+1]
			else:
				end_emission = start_emission
			increment = int(increments[p])
			for i in range(0, increment):
				p_co2_emission = start_emission + i * (end_emission-start_emission) / increment
				p_co2 = 0.71 * p_co2_emission # where are these numbers coming from?
				p_c = p_co2 / 3.67 
				add_p_ppm = subinterval_len * p_c / 2.13
				lsc = cls.lsc_p1 + cls.lsc_p2 * cum_sink
				absorbtion = 0.5 * cls.absorbtion_p1 * np.sign(ghg_level-lsc) * np.abs(ghg_level-lsc)**cls.absorbtion_p2
				cum_sink += absorbtion
				cum_forcing += cls.forcing_p1*np.sign(ghg_level-cls.forcing_p3)*np.abs(ghg_level-cls.forcing_p3)**cls.forcing_p2
				ghg_level += add_p_ppm - absorbtion

		if returning == "forcing":
			return cum_forcing
		elif returning == "ghg":
			return ghg_level
		else:
			raise ValueError("Does not recognize the returning string {}".format(returning))
	
	@classmethod
	def forcing_at_node(cls, m, node, tree, bau, subinterval_len):
		"""Calculates the forcing based mitigation leading up to the 
		damage calculation in `node`.

		Parameters
		----------
		m : ndarray 
			array of mitigations in each node. 
		node : int 
			the node for which the forcing is being calculated.

		Returns
		-------
		float 
			forcing 

		"""

		return cls._forcing_and_ghg_at_node(m, node, tree, bau, subinterval_len, returning="forcing")

	@classmethod
	def ghg_level_at_node(cls, m, node, tree, bau, subinterval_len):
		"""Calculates the GHG level leading up to the damage calculation in `node`.

		Parameters
		----------
		m : ndarray 
			array of mitigations in each node. 
		node : int 
			the node for which the GHG level is being calculated.

		Returns
		-------
		float 
			GHG level at node

		"""
		return cls._forcing_and_ghg_at_node(m, node,tree, bau, subinterval_len, returning="ghg")
		
