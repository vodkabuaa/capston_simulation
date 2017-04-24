import dlw

t = dlw.tree.TreeModel(decision_times=[0, 15, 45, 85, 185, 285, 385], prob_scale=1.0)
bau_default_model = dlw.bau.DLWBusinessAsUsual()
bau_default_model.bau_emissions_setup(t)
c = dlw.cost.DLWCost(t, bau_default_model.emit_level[0], g=92.08, a=3.413, join_price=2000.0, max_price=2500.0,
			tech_const=1.5, tech_scale=0.0, cons_at_0=30460.0)
df = dlw.damage.DLWDamage(tree=t, bau=bau_default_model, cons_growth=0.015, ghg_levels=[450, 650, 1000])
#df.damage_simulation(draws=4000000, peak_temp=6.0, disaster_tail=18.0, tip_on=True, 
#		temp_map=1, temp_dist_params=None, maxh=100.0, cons_growth=0.015)
df.import_damages()
df.forcing_init(sink_start=35.596, forcing_start=4.926, ghg_start=400, partition_interval=5,
	forcing_p1=0.13173, forcing_p2=0.607773, forcing_p3=315.3785, absorbtion_p1=0.94835,
	absorbtion_p2=0.741547, lsc_p1=285.6268, lsc_p2=0.88414)


m = np.array([0.70127532,0.88237503,0.67008528,1.0560499, 0.97725033,0.99704736,
			  0.54322163,1.16799415,1.12338119,1.15598646,1.04917023,1.20250835,
			  0.95369316,0.75708963,0.40455469,0.97771908,0.95193852,1.04690494,
			  1.02488011,1.05592216,1.02911578,1.17137021,1.1423471, 1.03523072,
			  1.0617091, 1.30045254,1.34396322,1.57874767,1.09461517,0.78595818,
			  0.58488512,1.01330241,1.04615827,1.03763915,1.0621624, 0.96794989,
			  0.98165542,1.03694265,1.08108007,0.96499028,0.96687322,1.00209777,
			  1.11069151,0.96085488,0.91191391,1.0389178, 1.19585869,0.97624981,
			  1.02156012,1.30190345,0.84209964,0.94240239,0.96689163,0.21734868,
			  0.72908764,1.10545033,1.91870387,1.40098805,1.56306369,1.41810261,
			  0.69188149,1.47006041,0.86803083])

u = dlw.utility.EZUtility(tree=t, damage=df, cost=c, period_len=5.0)
"""

ga_model = dlw.optimization.GenericAlgorithm(pop_amount=250, num_generations=250, cx_prob=0.8, mut_prob=0.5, 
							bound=3.0, num_feature=63, utility=u, print_progress=True)

gs_model = dlw.optimization.GradientSearch(learning_rate=0.01, var_nums=63, utility=u, accuracy=1e-7, 
						  iterations=100, print_progress=True)
final_pop, fitness = ga_model.run()
sort_pop = final_pop[np.argsort(fitness)][::-1]
print sort_pop[0]
m_opt, u_opt = gs_model.run(initial_point_list=sort_pop, topk=4)

utility_t, cons_t, cost_t, ce_t = u.utility(m_opt, return_trees=True)
dlw.tools.save_output(m_opt, u, utility_t, cons_t, cost_t, ce_t)
delta_cons_tree, delta_cost_array = dlw.tools.delta_consumption(m_opt, u, cons_t, cost_t, 0.01)
dlw.tools.save_sensitivity_analysis(m_opt, u, utility_t, cons_t, cost_t, ce_t, delta_cons_tree, delta_cost_array)


# Constraint first period mitigation to 0.0
#cfp_m, cfp_cons_tree, cfp_cost_array = dlw.tools.constraint_first_period(m, u, 0.0)
#cfp_utility_t, cfp_cons_t, cfp_cost_t, cfp_ce_t = u.utility(cfp_m, return_trees=True)
#dlw.tools.save_output(cfp_m, u, cfp_utility_t, cfp_cons_t, cfp_cost_t, cfp_ce_t)
#dlw.tools.save_sensitivity_analysis(cfp_m, u, cfp_utility_t, cfp_cons_t, cfp_cost_t, cfp_ce_t, 
#							  cfp_cons_tree, cfp_cost_array, "CFP")
"""