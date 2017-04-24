from __future__ import division, print_function
import numpy as np
from storage_tree import BigStorageTree, SmallStorageTree


def additional_ghg_emission(m, utility):
	"""Calculate the emission added by every node.

	Parameters
	----------
	m : ndarray or list
		array of mitigation
	utility : `Utility` object
		object of utility class
	
	Returns
	-------
	ndarray
		additional emission in nodes
	
	"""
	additional_emission = np.zeros(len(m))
	cache = set()
	for node in range(utility.tree.num_final_states, len(m)):
		path = utility.tree.get_path(node)
		for i in range(len(path)):
			if path[i] not in cache:
				additional_emission[path[i]] = (1.0 - m[path[i]]) *  utility.damage.bau.emission_to_ghg[i]
				cache.add(path[i])
	return additional_emission

def store_trees(prefix=None, start_year=2015, **kwargs):
	"""Saves values of `BaseStorageTree` objects. The file is saved into the 'data' directory
	in the current working directory. If there is no 'data' directory, one is created. 

	Parameters
	----------
	prefix : str, optional 
		prefix to be added to file_name
	start_year : int, optional
		start year of analysis
	**kwargs 
		arbitrary keyword arguments of `BaseStorageTree` objects

	"""
	if prefix is None:
		prefix = ""
	for name, tree in kwargs.items():
		tree.write_columns(prefix + "trees", name, start_year)

def delta_consumption(m, utility, cons_tree, cost_tree, delta_m):
	"""Calculate the changes in consumption and the mitigation cost component 
	of consumption when increaseing period 0 mitigiation with `delta_m`.

	Parameters
	----------
	m : ndarray or list
		array of mitigation
	utility : `Utility` object
		object of utility class
	cons_tree : `BigStorageTree` object
		consumption storage tree of consumption values
		from optimal mitigation values
	cost_tree : `SmallStorageTree` object
		cost storage tree of cost values from optimal mitigation values
	delta_m : float 
		value to increase period 0 mitigation by
	
	Returns
	-------
	tuple
		(storage tree of changes in consumption, ndarray of costs in first sub periods)

	"""
	m_copy = m.copy()
	m_copy[0] += delta_m

	new_utility_tree, new_cons_tree, new_cost_tree, new_ce_tree = utility.utility(m_copy, return_trees=True)

	for period in new_cons_tree.periods:
		new_cons_tree.tree[period] = (new_cons_tree.tree[period]-cons_tree.tree[period]) / delta_m

	first_period_intervals = new_cons_tree.first_period_intervals
	cost_array = np.zeros((first_period_intervals, 2))
	for i in range(first_period_intervals):
		potential_consumption = (1.0 + utility.cons_growth)**(new_cons_tree.subinterval_len * i)
		cost_array[i, 0] = potential_consumption * cost_tree[0]
		cost_array[i, 1] = (potential_consumption * new_cost_tree[0] - cost_array[i, 0]) / delta_m
	
	return new_cons_tree, cost_array, new_utility_tree[0]

def constraint_first_period(utility, first_node, m_size):
	"""Calculate the changes in consumption, the mitigation cost component of consumption,
	and new mitigation values when constraining the first period mitigation to `first_node`.

	Parameters
	----------
	m : ndarray or list
		array of mitigation
	utility : `Utility` object
		object of utility class
	first_node : float
		value to constrain first period to
	
	Returns
	-------
	tuple
		(new mitigation array, storage tree of changes in consumption, ndarray of costs in first sub periods)

	"""
	from optimization import GeneticAlgorithm, GradientSearch
	fixed_values = np.array([first_node])
	fixed_indicies = np.array([0])
	ga_model = GeneticAlgorithm(pop_amount=150, num_generations=150, cx_prob=0.8, mut_prob=0.5, bound=3.0,
								num_feature=m_size, utility=utility, fixed_values=fixed_values, 
								fixed_indicies=fixed_indicies, print_progress=True)

	gs_model = GradientSearch(learning_rate=0.001, var_nums=m_size, utility=utility, accuracy=1e-7,
							  iterations=250, fixed_values=fixed_values, fixed_indicies=fixed_indicies, 
							  print_progress=True)

	final_pop, fitness = ga_model.run()
	sort_pop = final_pop[np.argsort(fitness)][::-1]
	new_m, new_utility = gs_model.run(initial_point_list=sort_pop, topk=1)

	return new_m

def numerical_scc(m, utility, delta_m):
	utility_t, cons_t, cost_t, ce_t = utility.utility(m, return_trees=True)
	m_copy = m.copy()
	m_copy[0] += delta_m
	delta_utility_t, delta_cons_t, delta_cost_t, delta_ce_t = utility.utility(m_copy, return_trees=True)
	delta_utility = (delta_utility_t[0]-utility_t[0])
	node_eps = BigStorageTree(5.0, [0, 15, 45, 85, 185, 285, 385])
	scc = 0
	for period in cons_t.decision_times[1:]:
		cons_t.tree[period] = (delta_cons_t[period]-cons_t[period])
		for node in range(len(cons_t[period])):
			node_eps.tree[period][node] = delta_m
			adj_utiity = utility.adjusted_utility(m, node_cons_eps=node_eps)
			node_eps.tree[period][node] = 0.0
			cons_t.tree[period][node] = (cons_t[period][node]/(delta_m)) * ((adj_utiity-utility_t[0])/cons_t[period][node])
			cons_t.tree[period][node] = np.nan_to_num(cons_t[period][node])
		scc += cons_t.tree[period].sum()
	scc = scc*((delta_cons_t[0]-cons_t[0])/delta_utility)*delta_m*m[0]
	return scc


def find_ir(m, utility, payment, a=0.0, b=1.0): 
	"""Find the price of a bond that creates equal utility at time 0 as adding `payment` to the value of 
	consumption in the final period. The purpose of this function is to find the interest rate 
	embedded in the `EZUtility` model. 

	Parameters
	----------
	m : ndarray or list
		array of mitigation
	utility : `Utility` object
		object of utility class
	payment : float
		value added to consumption in the final period
	a : float, optional
		initial guess
	b : float, optional
		initial guess - f(b) needs to give different sign than f(a)
	
	Returns
	-------
	tuple
		result of optimization

	Note
	----
	requires the 'scipy' package

	"""
	from scipy.optimize import brentq

	def min_func(price):
		utility_with_final_payment = utility.adjusted_utility(m, final_cons_eps=payment)
		first_period_eps = payment * price
		utility_with_initial_payment = utility.adjusted_utility(m, first_period_consadj=first_period_eps)
		return utility_with_final_payment - utility_with_initial_payment

	return brentq(min_func, a, b)

def find_term_structure(m, utility, payment, a=0.0, b=1.5): 
	"""Find the price of a bond that creates equal utility at time 0 as adding `payment` to the value of 
	consumption in the final period. The purpose of this function is to find the interest rate 
	embedded in the `EZUtility` model. 

	Parameters
	----------
	m : ndarray or list
		array of mitigation
	utility : `Utility` object
		object of utility class
	payment : float
		value added to consumption in the final period
	a : float, optional
		initial guess
	b : float, optional
		initial guess - f(b) needs to give different sign than f(a)
	
	Returns
	-------
	tuple
		result of optimization

	Note
	----
	requires the 'scipy' package

	"""
	from scipy.optimize import brentq
	def min_func(price):
		period_cons_eps = np.zeros(int(utility.decision_times[-1]/utility.period_len) + 1)
		period_cons_eps[-2] = payment
		utility_with_payment = utility.adjusted_utility(m, period_cons_eps=period_cons_eps)
		first_period_eps = payment * price
		utility_with_initial_payment = utility.adjusted_utility(m, first_period_consadj=first_period_eps)
		return  utility_with_payment - utility_with_initial_payment
	try:
		return brentq(min_func, a, b)
	except:
		return 1e-11

def find_bec(m, utility, constraint_cost, a=-0.1, b=1.0):
	"""Used to find a value for consumption that equalizes utility at time 0 in two different solutions.

	Parameters
	----------
	m : ndarray or list
		array of mitigation
	utility : `Utility` object
		object of utility class
	constraint_cost : float
		utility cost of constraining period 0 to zero
	a : float, optional
		initial guess
	b : float, optional
		initial guess - f(b) needs to give different sign than f(a)
	
	Returns
	-------
	tuple
		result of optimization

	Note
	----
	requires the 'scipy' package

	"""
	from scipy.optimize import brentq

	def min_func(delta_con):
		base_utility = utility.adjusted_utility(m)
		new_utility = utility.adjusted_utility(m, first_period_consadj=delta_con)
		return new_utility - base_utility - constraint_cost

	return brentq(min_func, a, b)

def perpetuity_yield(price, start_date, a=0.1, b=10.0):
	"""Find the yield of a perpetuity starting at year `start_date`.

	Parameters
	----------
	price : float
		price of bond ending at `start_date`
	start_date : int
		start year of perpetuity
	a : float, optional
		initial guess
	b : float, optional
		initial guess - f(b) needs to give different sign than f(a)
	
	Returns
	-------
	tuple
		result of optimization

	Note
	----
	requires the 'scipy' package

	"""
	from scipy.optimize import brentq

	def min_func(perp_yield):
		return price - (100. / (perp_yield+100.))**start_date * (perp_yield + 100)/perp_yield

	try:
		return brentq(min_func, a, b)
	except:
		return 1e-11


def save_output(m, utility, utility_tree, cons_tree, cost_tree, ce_tree, prefix=None):
	"""Save the result of optimization and calculated values based on optimal mitigation. For every node the 
	function calculates and saves:
		
		* average mitigation
		* average emission
		* GHG level 
		* SCC 

	into the file `prefix` + 'node_period_output' in the 'data' directory in the current working directory. 

	For every period the function calculates and appends:
		
		* expected SCC/price
		* expected mitigation 
		* expected emission 
	
	into the file  `prefix` + 'node_period_output' in the 'data' directory in the current working directory. 

	The function also saves the values stored in the `BaseStorageTree` object parameters to a file called 
	`prefix` + 'tree' in the 'data' directory in the current working directory. If there is no 'data' 
	directory, one is created. 

	Parameters
	----------
	m : ndarray or list
		array of mitigation
	utility : `Utility` object
		object of utility class
	utility_tree : `BigStorageTree` object
		utility values from optimal mitigation values
	cons_tree : `BigStorageTree` object
		consumption values from optimal mitigation values
	cost_tree : `SmallStorageTree` object
		cost values from optimal mitigation values
	ce_tree : `BigStorageTree` object
		certain equivalence values from optimal mitigation values
	prefix : str, optional
		prefix to be added to file_name

	"""
	from tools import write_columns_csv, append_to_existing
	bau = utility.damage.bau
	tree = utility.tree
	periods = tree.num_periods
	prices = np.zeros(len(m))
	ave_mitigations = np.zeros(len(m))
	ave_emissions = np.zeros(len(m))
	expected_period_price = np.zeros(periods)
	expected_period_mitigation = np.zeros(periods)
	expected_period_emissions = np.zeros(periods)
	additional_emissions = additional_ghg_emission(m, utility)
	ghg_levels = utility.damage.ghg_level(m)

	periods = tree.num_periods
	for period in range(0, periods):
		years = tree.decision_times[period]
		period_years = tree.decision_times[period+1] - tree.decision_times[period]
		nodes = tree.get_nodes_in_period(period)
		num_nodes_period = 1 + nodes[1] - nodes[0]
		period_lens = tree.decision_times[:period+1] 
		for node in range(nodes[0], nodes[1]+1):
			path = np.array(tree.get_path(node, period))
			new_m = m[path]
			mean_mitigation = np.dot(new_m, period_lens) / years
			price = utility.cost.price(years, m[node], mean_mitigation)
			prices[node] = price
			ave_mitigations[node] = utility.damage.average_mitigation_node(m, node, period)
			ave_emissions[node] = additional_emissions[node] / (period_years*bau.emission_to_bau)
		probs = tree.get_probs_in_period(period)
		expected_period_price[period] = np.dot(prices[nodes[0]:nodes[1]+1], probs)
		expected_period_mitigation[period] = np.dot(ave_mitigations[nodes[0]:nodes[1]+1], probs)
		expected_period_emissions[period] = np.dot(ave_emissions[nodes[0]:nodes[1]+1], probs)

	if prefix is not None:
		prefix += "_" 
	else:
		prefix = ""

	write_columns_csv([m, prices, ave_mitigations, ave_emissions, ghg_levels], prefix+"node_period_output",
					   ["Node", "Mitigation", "Prices", "Average Mitigation", "Average Emission", "GHG Level"], [range(len(m))])

	append_to_existing([expected_period_price, expected_period_mitigation, expected_period_emissions],
						prefix+"node_period_output", header=["Period", "Expected Price", "Expected Mitigation",
						"Expected Emission"], index=[range(periods)], start_char='\n')

	store_trees(prefix=prefix, Utility=utility_tree, Consumption=cons_tree, 
				Cost=cost_tree, CertainEquivalence=ce_tree)

	
def save_sensitivity_analysis(m, utility, utility_tree, cons_tree, cost_tree, ce_tree, prefix=None, return_delta_utility=False):
	"""Calculate and save sensitivity analysis based on the optimal mitigation. For every sub-period, i.e. the 
	periods given by the utility calculations, the function calculates and saves:
		
		* discount prices
		* net expected damages
		* expected damages
		* risk premium
		* expected SDF
		* cross SDF & damages
		* discounted expected damages
		* cov term
		* scaled net expected damages
		* scaled risk premiums
	
	into the file  `prefix` + 'sensitivity_output' in the 'data' directory in the current working directory. 

	Furthermore, for every node the function calculates and saves:
	
		* SDF 
		* delta consumption
		* forward marginal utility  
		* up-node marginal utility
		* down-node marginal utility
	
	into the file `prefix` + 'tree' in the 'data' directory in the current working directory. If there is no 'data' 
	directory, one is created. 

	Parameters
	----------
	m : ndarray or list
		array of mitigation
	utility : `Utility` object
		object of utility class
	utility_tree : `BigStorageTree` object
		utility values from optimal mitigation values
	cons_tree : `BigStorageTree` object
		consumption values from optimal mitigation values
	cost_tree : `SmallStorageTree` object
		cost values from optimal mitigation values
	ce_tree : `BigStorageTree` object
		certain equivalence values from optimal mitigation values
	prefix : str, optional
		prefix to be added to file_name

	"""
	from tools import write_columns_csv, append_to_existing

	sdf_tree = BigStorageTree(utility.period_len, utility.decision_times)
	sdf_tree.set_value(0, np.array([1.0]))

	discount_prices = np.zeros(len(sdf_tree))
	net_expected_damages = np.zeros(len(sdf_tree))
	expected_damages = np.zeros(len(sdf_tree))
	risk_premiums = np.zeros(len(sdf_tree))
	expected_sdf = np.zeros(len(sdf_tree))
	cross_sdf_damages = np.zeros(len(sdf_tree))
	discounted_expected_damages = np.zeros(len(sdf_tree))
	net_discount_damages = np.zeros(len(sdf_tree))
	cov_term = np.zeros(len(sdf_tree))

	discount_prices[0] = 1.0
	cost_sum = 0

	end_price = find_term_structure(m, utility, 0.01)
	perp_yield = perpetuity_yield(end_price, sdf_tree.periods[-2])

	delta_cons_tree, delta_cost_array, delta_utility = delta_consumption(m, utility, cons_tree, cost_tree, 0.01)
	mu_0, mu_1, mu_2 = utility.marginal_utility(m, utility_tree, cons_tree, cost_tree, ce_tree)
	sub_len = sdf_tree.subinterval_len
	i = 1
	for period in sdf_tree.periods[1:]:
		node_period = sdf_tree.decision_interval(period)
		period_probs = utility.tree.get_probs_in_period(node_period)
		expected_damage = np.dot(delta_cons_tree[period], period_probs)
		expected_damages[i] = expected_damage
		
		if sdf_tree.is_information_period(period-sdf_tree.subinterval_len):
			total_probs = period_probs[::2] + period_probs[1::2]
			mu_temp = np.zeros(2*len(mu_1[period-sub_len]))
			mu_temp[::2] = mu_1[period-sub_len]
			mu_temp[1::2] = mu_2[period-sub_len]
			sdf = (np.repeat(total_probs, 2) / period_probs) * (mu_temp/np.repeat(mu_0[period-sub_len], 2))
			period_sdf = np.repeat(sdf_tree.tree[period-sub_len],2)*sdf 
		else:
			sdf = mu_1[period-sub_len]/mu_0[period-sub_len]
			period_sdf = sdf_tree[period-sub_len]*sdf 

		expected_sdf[i] = np.dot(period_sdf, period_probs)
		cross_sdf_damages[i] = np.dot(period_sdf, delta_cons_tree[period]*period_probs)
		cov_term[i] = cross_sdf_damages[i] - expected_sdf[i]*expected_damage

		discount_prices[i] = expected_sdf[i]
		sdf_tree.set_value(period, period_sdf)

		if i < len(delta_cost_array):
			net_discount_damages[i] = -(expected_damage + delta_cost_array[i, 1]) * expected_sdf[i] / delta_cons_tree[0]
			cost_sum += -delta_cost_array[i, 1] * expected_sdf[i] / delta_cons_tree[0]
		else:
			net_discount_damages[i] = -expected_damage * expected_sdf[i] / delta_cons_tree[0]

		risk_premiums[i] = -cov_term[i]/delta_cons_tree[0]
		discounted_expected_damages[i] = -expected_damage * expected_sdf[i] / delta_cons_tree[0]
		i += 1

	damage_scale = utility.cost.price(0, m[0], 0) / (net_discount_damages.sum()+risk_premiums.sum())
	scaled_discounted_ed = net_discount_damages * damage_scale
	scaled_risk_premiums = risk_premiums * damage_scale

	if prefix is not None:
		prefix += "_" 
	else:
		prefix = ""

	write_columns_csv([discount_prices, net_discount_damages, expected_damages, risk_premiums, expected_sdf, cross_sdf_damages, 
					   discounted_expected_damages, cov_term, scaled_discounted_ed, scaled_risk_premiums], prefix + "sensitivity_output",
					   ["Year", "Discount Prices", "Net Expected Damages", "Expected Damages", "Risk Premium",
					    "Expected SDF", "Cross SDF & Damages", "Discounted Expected Damages", "Cov Term", "Scaled Net Expected Damages",
					    "Scaled Risk Premiums"], [sdf_tree.periods.astype(int)+2015]) 

	append_to_existing([[end_price], [perp_yield], [scaled_discounted_ed.sum()], [scaled_risk_premiums.sum()], [utility.cost.price(0, m[0], 0)],
						cost_sum], prefix+"sensitivity_output", header=["Zero Bound Price", "Perp Yield", "Expected Damages", "Risk Premium", 
						"SCC", "Sum Delta Cost"], start_char='\n')
	
	store_trees(prefix=prefix, SDF=sdf_tree, DeltaConsumption=delta_cons_tree, MU_0=mu_0, MU_1=mu_1, MU_2=mu_2)

	if return_delta_utility:
		return delta_utility


def save_constraint_analysis(cfp_m, utility, delta_util_x, delta_cons=0.01, prefix=None):
	from tools import write_columns_csv
	utility_given_delta_con = utility.adjusted_utility(cfp_m, first_period_consadj=delta_cons)
	delta_util_c = utility_given_delta_con - utility.utility(cfp_m)
	delta_con = find_bec(cfp_m, utility, delta_util_x)
	marginal_benefit = (delta_util_x / delta_util_c ) * delta_con * utility.cost.cons_per_ton / delta_cons
	delta_cons_billions = delta_con * utility.cost.cons_per_ton * utility.damage.bau.emit_level[0]
	delta_emission_gton = delta_cons * utility.damage.bau.emit_level[0]
	deadweight = delta_con * utility.cost.cons_per_ton / delta_cons
	if prefix is not None:
		prefix += "_" 
	else:
		prefix = ""
	write_columns_csv([delta_util_x, delta_util_c, [delta_con], marginal_benefit, [delta_cons_billions],
					   [delta_emission_gton], [deadweight]], prefix+"constraint_output",
					   header=["Delta Utility Mitigation", "Delta Utility Consumption", "Delta Consumption", 
					   "Marginal Benefit", "Delta Consumption Billions", "Delta Emission GTon", "Deadweight"])



	