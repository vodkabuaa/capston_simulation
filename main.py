
from tree import TreeModel
from bau import DLWBusinessAsUsual
from cost import DLWCost
from damage import DLWDamage
from utility import EZUtility
from analysis import *
from tools import *
from optimization import *


if __name__ == "__main__":
	header, indices, data = import_csv("DLW_research_runs", indices=2)
	_1, _2, base_data = import_csv("base_case_node_period_output", indices=1) #saved base_case output
	m_opt = base_data[:,0]

	for i in range(17, 18):
		name = indices[i][1]
		a, ra, eis, pref, temp, tail, growth, tech_chg, tech_scale, joinp, maxp, on, maps = data[i]
		print(name, ra, eis)
		if on == 1.0:
			on = True
		else:
			on = False
		maps = int(maps)
		t = TreeModel(decision_times=[0, 15, 45, 85, 185, 285, 385], prob_scale=1.0)
		bau_default_model = DLWBusinessAsUsual()
		bau_default_model.bau_emissions_setup(t)
		c = DLWCost(t, bau_default_model.emit_level[0], g=92.08, a=3.413, join_price=joinp, max_price=maxp,
					tech_const=tech_chg, tech_scale=tech_scale, cons_at_0=30460.0)

		df = DLWDamage(tree=t, bau=bau_default_model, cons_growth=growth, ghg_levels=[450, 650, 1000], subinterval_len=5)
		#df.damage_simulation(draws=4000000, peak_temp=temp, disaster_tail=tail, tip_on=on,
		#					 temp_map=maps, temp_dist_params=None, maxh=100.0, cons_growth=growth)
		df.import_damages()

		u = EZUtility(tree=t, damage=df, cost=c, period_len=5.0, eis=eis, ra=ra, time_pref=pref)

		if a <= 2.0:
			ga_model = GeneticAlgorithm(pop_amount=150, num_generations=100, cx_prob=0.8, mut_prob=0.5,
								bound=1.5, num_feature=63, utility=u, print_progress=True)

			gs_model = GradientSearch(learning_rate=0.0001, var_nums=63, utility=u, accuracy=1e-8,
							  iterations=200, print_progress=True)
			final_pop, fitness = ga_model.run()
			sort_pop = final_pop[np.argsort(fitness)][::-1]
			m_opt, u_opt = gs_model.run(initial_point_list=sort_pop, topk=3)
			numpy.savetxt("fitness.csv", fitness, delimiter=",")
			numpy.savetxt("u_opt.csv", u_opt, delimiter=",")
			#m_opt, u_opt = gs_model.run(initial_point_list=[m_opt], topk=1)
			utility_t, cons_t, cost_t, ce_t = u.utility(m_opt, return_trees=True)
			save_output(m_opt, u, utility_t, cons_t, cost_t, ce_t, prefix="gs3_"+name)
			save_sensitivity_analysis(m_opt, u, utility_t, cons_t, cost_t, ce_t, prefix="gs3_"+name)


		# Constraint first period mitigation to 0.0
		else:
			cfp_m = constraint_first_period(u, 0.0, t.num_decision_nodes)
			cfp_utility_t, cfp_cons_t, cfp_cost_t, cfp_ce_t = u.utility(cfp_m, return_trees=True)
			save_output(cfp_m, u, cfp_utility_t, cfp_cons_t, cfp_cost_t, cfp_ce_t, prefix="CFP_"+name)
			delta_utility = save_sensitivity_analysis(cfp_m, u, cfp_utility_t, cfp_cons_t, cfp_cost_t, cfp_ce_t,
												    "CFP_"+name, return_delta_utility=True)
			delta_utility_x = delta_utility - cfp_utility_t[0]
			save_constraint_analysis(cfp_m, u, delta_utility_x, prefix="CFP_"+name)
