from __future__ import division, print_function
from abc import ABCMeta, abstractmethod
import numpy as np
from storage_tree import BigStorageTree, SmallStorageTree

np.seterr(all='ignore')

class EZUtility(object):
	"""Calculation of Epstein-Zin utility for the EZ-Climate model. 

	The Epstein-Zin utility allows for different rates of substitution across time and 
	states. For specification see DLW-paper.

	If `add_penalty_cost` is True the model will calculate a penalty cost for GHG levels 
	below 280. How large this penalty becomes is controlled using the parameters `max_penalty` 
	and `penalty_scale`.

	Parameters
	----------
	tree : `TreeModel` object 
		tree structure used
	damage : `Damage` object
		class that provides damage methods
	cost : `Cost` object
		class that procides cost methods
	period_len : float
		subinterval length
	eis : float, optional
		elasticity of intertemporal substitution
	ra : float, optional
		risk-aversion
	time_pref : float, optional
		pure rate of time preference
	add_penalty_cost : bool, optional
		True if utility calculation should include penalty cost for 
		GHG levels below 280, else False
	max_penalty : float, optional
		maximum penalty that can be added to penalty cost for one period	
	penalty_scale : float, optional
		constant that is multiplied to the penalty cost

	Attributes
	----------
	tree : `TreeModel` object 
		tree structure used
	damage : `Damage` object
		class that provides damage methods
	cost : `Cost` object
		class that procides cost methods
	period_len : float
		subinterval length
	decision_times : ndarray 
		years in the future where decisions will be made
	cons_growth : float
		consumption growth 
	growth_term : float
		1 + cons_growth
	r : float
		the parameter rho from the DLW-paper
	a : float
		the parameter alpha from the DLW-paper
	b : float
		the parameter beta from the DLW-paper
	add_penalty_cost : bool, optional
		True if utility calculation should include penalty cost for 
		GHG levels below 280, else False
	max_penalty : float
		maximum penalty that can be added to penalty cost for one period	
	penalty_scale : float
		constant that is multiplied to the penalty cost

	"""

	def __init__(self, tree, damage, cost, period_len, eis=0.9, ra=7.0, time_pref=0.005, 
		         add_penalty_cost=False, max_penalty=0.0, penalty_scale=1.0):
		self.tree = tree
		self.damage = damage
		self.cost = cost
		self.period_len = period_len
		self.decision_times = tree.decision_times
		self.cons_growth = damage.cons_growth
		self.growth_term = 1.0 + self.cons_growth
		self.r = 1.0 - 1.0/eis
		self.a = 1.0 - ra
		self.b = (1.0-time_pref)**period_len
		self.potential_cons = np.ones(self.decision_times.shape) + self.cons_growth
		self.potential_cons = self.potential_cons ** self.decision_times
		self.add_penalty_cost = add_penalty_cost
		self.max_penalty = max_penalty
		self.penalty_scale = penalty_scale

	def _end_period_utility(self, m, utility_tree, cons_tree, cost_tree, penalty_cost):
		"""Calculate the terminal utility."""
		period_ave_mitigation = self.damage.average_mitigation(m, self.tree.num_periods)
		period_damage = self.damage.damage_function(m, self.tree.num_periods)
		damage_nodes = self.tree.get_nodes_in_period(self.tree.num_periods)
		
		period_mitigation = m[damage_nodes[0]:damage_nodes[1]+1]
		period_cost = self.cost.cost(self.tree.num_periods, period_mitigation, period_ave_mitigation)

		continuation = (1.0 / (1.0 - self.b*(self.growth_term**self.r)))**(1.0/self.r)

		cost_tree.set_value(cost_tree.last_period, period_cost)
		cons_tree.set_value(cons_tree.last_period, self.potential_cons[-1] * (1.0 - period_damage))
		utility_tree.set_value(utility_tree.last_period, (1.0 - self.b)**(1.0/self.r) * cons_tree.last * continuation)

	def _end_period_marginal_utility(self, mu_tree_0, mu_tree_1, ce_tree, utility_tree, cons_tree):
		"""Calculate the terminal marginal utility."""
		ce_term = utility_tree.last**self.r - (1.0 - self.b)*cons_tree.last**self.r
		ce_tree.set_value(ce_tree.last_period, ce_term)

		mu_0_last = (1.0 - self.b)*(utility_tree[utility_tree.last_period-self.period_len] / cons_tree.last)**(1.0-self.r)
		mu_tree_0.set_value(mu_tree_0.last_period, mu_0_last)
		mu_0 = self._mu_0(cons_tree[cons_tree.last_period-self.period_len], ce_tree[ce_tree.last_period-self.period_len])
		mu_tree_0.set_value(mu_tree_0.last_period-self.period_len, mu_0)

		next_term = self.b * (1.0 - self.b) / (1.0 - self.b * self.growth_term**self.r)
		mu_1 = utility_tree[utility_tree.last_period-self.period_len]**(1-self.r) * next_term * cons_tree.last**(self.r-1.0)
		mu_tree_1.set_value(mu_tree_1.last_period-self.period_len, mu_1)
		

	def _certain_equivalence(self, period, damage_period, utility_tree):
		"""Caclulate certainty equivalence utility. If we are between decision nodes, i.e. no branching,
		then certainty equivalent utility at time period depends only on the utility next period 
		given information known today. Otherwise the certainty equivalent utility is the ability 
		weighted sum of next period utility over the partition reachable from the state.
		"""
		if utility_tree.is_information_period(period):
			damage_nodes = self.tree.get_nodes_in_period(damage_period+1)
			probs = self.tree.node_prob[damage_nodes[0]:damage_nodes[1]+1]
			even_probs = probs[::2]
			odd_probs = probs[1::2]
			even_util = ((utility_tree.get_next_period_array(period)[::2])**self.a) * even_probs
			odd_util = ((utility_tree.get_next_period_array(period)[1::2])**self.a) * odd_probs
			ave_util = (even_util + odd_util) / (even_probs + odd_probs)
			cert_equiv = ave_util**(1.0/self.a)
		else:
			# no branching implies certainty equivalent utility at time period depends only on
			# the utility next period given information known today
			cert_equiv = utility_tree.get_next_period_array(period)
		return cert_equiv


	def _interval_penalty(self, period_ghg):
		"""Penalty cost function."""
		return np.maximum(0.0, np.minimum(np.sign(period_ghg)*((280.0-period_ghg)/period_ghg), self.max_penalty))
		#return self.max_penalty/(1+100*np.exp(0.10*(period_ghg-280)))

	def penalty_cost(self, m):
		"""Calculate and return the penalty cost of GHG levels below 280. The penalties in previous nodes
		in the path leading to the current node is summed and added to current period's penalty, given by

			max(0, min((280-GHG level)/GHG level, max_penalty)

		The method returns a `BigStorageTree` object with penalties for every period where utilities are calculated. 

		Parameters
		----------
		m : ndarray or list
			array of mitigation
		
		Returns
		-------
		`BigStorageTree`
			penalties in every period where utilities are calculated

		"""
		penalty_cost = BigStorageTree(self.period_len, self.decision_times)
		if not self.add_penalty_cost:
			return penalty_cost

		ghg_levels = self.damage.ghg_level(m, periods=self.tree.num_periods)
		interval_length = self.decision_times[1:] - self.decision_times[:-1] 

		sum_size = 1
		prev_ghg_level = ghg_levels[0]

		for i in range(1, len(self.tree.decision_times)):
			time_period = self.tree.decision_times[i]
			prev_time_period = self.tree.decision_times[i-1]
			prev_penalty_cost = penalty_cost[prev_time_period]
			len_arr = len(penalty_cost[time_period])			

			ghg_level = ghg_levels[sum_size:sum_size+len_arr]
			increment = self.period_len
			total_increment = time_period-prev_time_period
			if prev_ghg_level.shape != ghg_level.shape:
				prev_ghg_level = np.repeat(prev_ghg_level, 2)
				prev_penalty_cost = np.repeat(prev_penalty_cost, 2)

			while prev_time_period < time_period:
				prev_time_period += self.period_len
				this_period_ghg = prev_ghg_level + (increment/total_increment) * (ghg_level-prev_ghg_level)
				this_period_ghg[this_period_ghg == 0] = 1.0 # doesn't really matter if we put 1.0 or a value closer to 0
				penalty_cost.set_value(prev_time_period, prev_penalty_cost + self._interval_penalty(this_period_ghg)/self.period_len)
				
				prev_penalty_cost = penalty_cost[prev_time_period]
				increment += self.period_len

			prev_ghg_level = ghg_level
			sum_size += len_arr

		return penalty_cost

	def _utility_generator(self, m, utility_tree, cons_tree, cost_tree, ce_tree, penalty_cost, cons_adj=0.0):
		"""Generator for calculating utility for each utility period besides the terminal utility."""
		periods = utility_tree.periods[::-1]

		for period in periods[1:]:
			damage_period = utility_tree.between_decision_times(period)
			cert_equiv = self._certain_equivalence(period, damage_period, utility_tree)

			if utility_tree.is_decision_period(period+self.period_len):
				damage_nodes = self.tree.get_nodes_in_period(damage_period)	
				period_mitigation =  m[damage_nodes[0]:damage_nodes[1]+1]
				period_ave_mitigation = self.damage.average_mitigation(m, damage_period)
				period_cost = self.cost.cost(damage_period, period_mitigation, period_ave_mitigation)
				period_damage = self.damage.damage_function(m, damage_period)
				cost_tree.set_value(cost_tree.index_below(period+self.period_len), period_cost)

			#period_consumption = self.potential_cons[damage_period] * (1.0 - period_damage) * (1.0 - period_cost)
		
			if not utility_tree.is_decision_period(period):

				next_consumption = cons_tree.get_next_period_array(period)
				segment = period - utility_tree.decision_times[damage_period]
				interval = segment + utility_tree.subinterval_len

				if utility_tree.is_decision_period(period+self.period_len):
					next_cost = cost_tree[period+self.period_len]
					if period < utility_tree.decision_times[-2]:
						next_consumption *= (1.0 - np.repeat(period_cost,2)) / (1.0 - next_cost)
						
				if period < utility_tree.decision_times[-2]:
					period_consumption = self.potential_cons[damage_period] * (1.0 - np.repeat(period_damage,2)) \
										 * (1.0 - np.repeat(period_cost,2) - self.penalty_scale*penalty_cost[period])
					#period_consumption = ((next_consumption/np.repeat(period_consumption,2))**(segment/float(interval))) * np.repeat(period_consumption,2)
				else:
					period_consumption = self.potential_cons[damage_period] * (1.0 - period_damage) \
					 					 * (1.0 - period_cost - self.penalty_scale*penalty_cost[period])
					#period_consumption = ((next_consumption/period_consumption)**(segment/float(interval)))*period_consumption
				
				period_consumption = ((next_consumption/period_consumption)**(segment/float(interval)))*period_consumption
			
			else:
				period_consumption = self.potential_cons[damage_period] * (1.0 - period_damage) \
									 * (1.0 - period_cost - self.penalty_scale*penalty_cost[period])
			
			if period == 0:
				period_consumption += cons_adj

			ce_term = self.b * cert_equiv**self.r
			ce_tree.set_value(period, ce_term)
			period_consumption[period_consumption < 0.0] = 1e-11
			period_consumption[np.isnan(period_consumption)] = 1e-11
			cons_tree.set_value(period, period_consumption)
			u = ((1.0-self.b)*period_consumption**self.r + ce_term)**(1.0/self.r)
			yield u, period

	def utility(self, m, return_trees=False):
		"""Calculating utility for the specific mitigation decisions `m`.

		Parameters
		----------
		m : ndarray or list
			array of mitigations
		return_trees : bool
			True if methid should return trees calculated in producing the utility

		Returns
		-------
		ndarray or tuple 
			tuple of `BaseStorageTree` if return_trees else ndarray with utility at period 0

		Examples:
		---------
		Assuming we have declared a EZUtility object as 'ezu' and have a mitigation array 'm'
		
		>>> ezu.utility(m)
		array([ 9.83391921])
		>>> utility_tree, cons_tree, cost_tree, ce_tree = ezu.utility(m, return_trees=True)

		"""
		utility_tree = BigStorageTree(subinterval_len=self.period_len, decision_times=self.decision_times)
		cons_tree = BigStorageTree(subinterval_len=self.period_len, decision_times=self.decision_times)
		ce_tree = BigStorageTree(subinterval_len=self.period_len, decision_times=self.decision_times)
		cost_tree = SmallStorageTree(decision_times=self.decision_times)
		penalty_cost = self.penalty_cost(m)

		self._end_period_utility(m, utility_tree, cons_tree, cost_tree, penalty_cost)
		it = self._utility_generator(m, utility_tree, cons_tree, cost_tree, ce_tree, penalty_cost)
		for u, period in it:
			utility_tree.set_value(period, u)
			
		if return_trees:
			return utility_tree, cons_tree, cost_tree, ce_tree
		return utility_tree[0]


	def adjusted_utility(self, m, period_cons_eps=None, node_cons_eps=None, final_cons_eps=0.0,
						 first_period_consadj=0.0, return_trees=False):
		"""Calculating adjusted utility for sensitivity analysis. Used e.g. to find zero-coupon bond price.
		Values in parameters are used to adjusted the utility in different ways.

		Parameters
		----------
		m : ndarray 
			array of mitigations
		period_cons_eps : ndarray, optional
			array of increases in consumption per period
		node_cons_eps : `SmallStorageTree`, optional 
			increases in consumption per node
		final_cons_eps : float, optional
			value to increase the final utilities by
		first_period_consadj : float, optional 
			value to increase consumption at period 0 by
		return_trees : bool, optional 
			True if method should return trees calculculated in producing the utility

		Returns
		-------
		ndarray or tuple 
			tuple of `BaseStorageTree` if return_trees else ndarray with utility at period 0

		Examples
		---------
		Assuming we have declared a EZUtility object as 'ezu' and have a mitigation array 'm'
		
		>>> ezu.adjusted_utility(m, final_cons_eps=0.1)
		array([ 9.83424045])
		>>> utility_tree, cons_tree, cost_tree, ce_tree = ezu.adjusted_utility(m, final_cons_eps=0.1, return_trees=True)
		
		>>> arr = np.zeros(int(ezu.decision_times[-1]/ezu.period_len) + 1)
		>>> arr[-1] = 0.1
		>>> ezu.adjusted_utility(m, period_cons_eps=arr)
		array([ 9.83424045])
		
		>>> bst = BigStorageTree(5.0, [0, 15, 45, 85, 185, 285, 385])
		>>> bst.set_value(bst.last_period, np.repeat(0.01, len(bst.last)))
		>>> ezu.adjusted_utility(m, node_cons_eps=bst)
		array([ 9.83391921])

		The last example differs from the rest in that the last values of the `node_cons_eps` will never be
		used. Hence if you want to update the last period consumption, use one of these two methods. 

		>>> ezu.adjusted_utility(m, first_period_consadj=0.01)
		array([ 9.84518772])

		"""
		utility_tree = BigStorageTree(subinterval_len=self.period_len, decision_times=self.decision_times)
		cons_tree = BigStorageTree(subinterval_len=self.period_len, decision_times=self.decision_times)
		ce_tree = BigStorageTree(subinterval_len=self.period_len, decision_times=self.decision_times)
		cost_tree = SmallStorageTree(decision_times=self.decision_times)

		periods = utility_tree.periods[::-1]
		if period_cons_eps is None:
			period_cons_eps = np.zeros(len(periods))
		if node_cons_eps is None:
			node_cons_eps = BigStorageTree(subinterval_len=self.period_len, decision_times=self.decision_times)
		
		penalty_cost = self.penalty_cost(m)
		self._end_period_utility(m, utility_tree, cons_tree, cost_tree, penalty_cost)

		it = self._utility_generator(m, utility_tree, cons_tree, cost_tree, ce_tree, penalty_cost, first_period_consadj)
		i = len(utility_tree)-2
		for u, period in it:
			if period == periods[1]:
				mu_0 = (1.0-self.b) * (u/cons_tree[period])**(1.0-self.r)
				next_term = self.b * (1.0-self.b) / (1.0-self.b*self.growth_term**self.r)
				mu_1 = (u**(1.0-self.r)) * next_term * (cons_tree.last**(self.r-1.0))
				u += (final_cons_eps+period_cons_eps[-1]+node_cons_eps.last) * mu_1
				u +=  (period_cons_eps[i]+node_cons_eps.tree[period]) * mu_0
				utility_tree.set_value(period, u)
			else:
				mu_0, m_1, m_2 = self._period_marginal_utility(mu_0, mu_1, m, period, utility_tree, cons_tree, ce_tree)
				u +=  (period_cons_eps[i] + node_cons_eps.tree[period])*mu_0
				utility_tree.set_value(period, u)
			i -= 1

		if return_trees:
			return utility_tree, cons_tree, cost_tree, ce_tree
		return utility_tree.tree[0]

	def _mu_0(self, cons, ce_term):
		"""Marginal utility with respect to consumption function."""
		t1 = (1.0 - self.b)*cons**(self.r-1.0)
		t2 = (ce_term - (self.b-1.0)*cons**self.r)**((1.0/self.r)-1.0)
		return t1 * t2

	def _mu_1(self, cons, prob, cons_1, cons_2, ce_1, ce_2, do_print=False):
		""" marginal utility with respect to consumption next period."""
		t1 = (1.0-self.b) * self.b * prob * cons_1**(self.r-1.0)
		t2 = (ce_1 - (self.b-1.0) * cons_1**self.r )**((self.a/self.r)-1)
		t3 = (prob * (ce_1 - (self.b*(cons_1**self.r)) + cons_1**self.r)**(self.a/self.r) \
			 + (1.0-prob) * (ce_2 - (self.b-1.0) * cons_2**self.r)**(self.a/self.r))**((self.r/self.a)-1.0)
		t4 = prob * (ce_1-self.b * (cons_1**self.r) + cons_1**self.r)**(self.a/self.r) \
			 + (1.0-prob) * (ce_2 - self.b * (cons_2**self.r) + cons_2**self.r)**(self.a/self.r)
		t5 = (self.b * t4**(self.r/self.a) - (self.b-1.0) * cons**self.r )**((1.0/self.r)-1.0)

		return t1 * t2 * t3 * t5 

	def _mu_2(self, cons, prev_cons, ce_term):
		"""Marginal utility with respect to last period consumption."""
		t1 = (1.0-self.b) * self.b * prev_cons**(self.r-1.0)
		t2 = ((1.0 - self.b) * cons**self.r - (self.b - 1.0) * self.b \
		     * prev_cons**self.r + self.b * ce_term)**((1.0/self.r)-1.0)
		return t1 * t2

	def _period_marginal_utility(self, prev_mu_0, prev_mu_1, m, period, utility_tree, cons_tree, ce_tree):
		"""Marginal utility for each node in a period."""
		damage_period = utility_tree.between_decision_times(period)
		mu_0 = self._mu_0(cons_tree[period], ce_tree[period])
		
		prev_ce = ce_tree.get_next_period_array(period)
		prev_cons = cons_tree.get_next_period_array(period)
		if utility_tree.is_information_period(period):
			probs = self.tree.get_probs_in_period(damage_period+1)
			up_prob = np.array([probs[i]/(probs[i]+probs[i+1]) for i in range(0, len(probs), 2)])
			down_prob = 1.0 - up_prob

			up_cons = prev_cons[::2]
			down_cons = prev_cons[1::2]
			up_ce = prev_ce[::2]
			down_ce = prev_ce[1::2]
			
			mu_1 = self._mu_1(cons_tree[period], up_prob, up_cons, down_cons, up_ce, down_ce)
			mu_2 = self._mu_1(cons_tree[period], down_prob, down_cons, up_cons, down_ce, up_ce)
			return mu_0, mu_1, mu_2 
		else:
			mu_1 = self._mu_2(cons_tree[period], prev_cons, prev_ce)
			return mu_0, mu_1, None 

	def marginal_utility(self, m, utility_tree, cons_tree, cost_tree, ce_tree):
		"""Calculating marginal utility for sensitivity analysis, e.g. in the SSC decomposition.

		Parameters
		----------
		m : ndarray
			array of mitigations
		utility_tree : `BigStorageTree` object 
			utility values from using mitigation `m`
		cons_tree : `BigStorageTree` object
			consumption values from using mitigation `m`
		cost_tree : `SmallStorageTree` object
			cost values from using mitigation `m`
		ce_tree : `BigStorageTree` object
			certain equivalence values from using mitigation `m`

		Returns
		-------
		tuple
			marginal utility tree

		Examples
		--------
		Assuming we have declared a EZUtility object as 'ezu' and have a mitigation array 'm'.
		>>> 
		>>> utility_tree, cons_tree, cost_tree, ce_tree = ezu.utility(m, return_trees=True)
		>>> mu_0_tree, mu_1_tree, mu_2_tree = ezu.marginal_utility(m, utility_tree, cons_tree, cost_tree, ce_tree)
		>>> mu_0_tree[0] # value at period 0
		array([ 0.33001256])
		>>> mu_1_tree[0] # value at period 0
		array([ 0.15691619])
		>>> mu_2_tree[0] # value at period 0
		array([ 0.13948175])

		"""
		#could add ce_tree to parameter list.
		mu_tree_0 = BigStorageTree(subinterval_len=self.period_len, decision_times=self.decision_times)
		mu_tree_1 = BigStorageTree(subinterval_len=self.period_len, decision_times=self.decision_times)
		mu_tree_2 = SmallStorageTree(decision_times=self.decision_times)
		
		self._end_period_marginal_utility(mu_tree_0, mu_tree_1, ce_tree, utility_tree, cons_tree)
		periods = utility_tree.periods[::-1]

		for period in periods[2:]:
			mu_0, mu_1, mu_2 = self._period_marginal_utility(mu_tree_0.get_next_period_array(period),
				mu_tree_1.get_next_period_array(period), m, period, utility_tree, cons_tree, ce_tree)
			mu_tree_0.set_value(period, mu_0)
			mu_tree_1.set_value(period, mu_1)
			if mu_2 is not None:
				mu_tree_2.set_value(period, mu_2)

		return mu_tree_0, mu_tree_1, mu_tree_2

	def partial_grad(self, m, i, delta=1e-8):
		"""Calculate the ith element of the gradient vector."""
		m_copy = m.copy()
		m_copy[i] -= delta
		minus_utility = self.utility(m_copy)
		m_copy[i] += 2*delta
		plus_utility = self.utility(m_copy)
		grad = (plus_utility-minus_utility) / (2*delta)
		return grad
