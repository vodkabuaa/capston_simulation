
from tree import TreeModel
from bau import DLWBusinessAsUsual
from cost import DLWCost
from damage import DLWDamage
from utility import EZUtility
from analysis import *
from tools import *
from optimization import *

m = np.array([0.6448154968019163, 0.84012434278802794, 0.6140597739129614,1.053213193089992,0.96107266489512033,
			  0.93702469418531187, 0.47904534182048503, 1.172028826946361, 1.1717093775960734, 1.2063669892276478, 
			  1.115639550950138, 1.2719447205297747, 0.87659457552391229, 0.72026153414981564, 0.45097772288264576,
			  1.0000963768516449, 1.0001434538607015, 1.0006285683942406, 1.0006557325888641, 1.0014139991174247, 
			  1.0013995462047791, 1.092207809897215, 1.0918228441229365, 1.0010103780026331, 1.0010372641995109,
			  1.3960886620314064, 1.4028751876578214, 1.4717954125266277, 0.99448984834194076, 0.87758251674791976,
			  0.55061314106605752, 0.99988485444369413, 0.9997605318990177, 0.99978165373701755, 0.99975520798513862, 
			  0.99970786449209093, 0.851482619622975, 0.99817545279313868, 0.99989801224204089, 0.99992778370793156,
			  0.99970369530305037, 0.99940400188033696, 1.2749045531176237, 0.99967391105563108, 0.99937658445481747,
			  1.0002052763902012, 1.0007507177492088, 1.000327194690938, 0.99994431387011606, 0.99938155295069464,
			  0.99895955079531118, 1.0003145202626536, 1.0005685921723757, 0.98809520214972246, 0.88845504023002253,
			  1.3240632538580832, 1.3718722934631753, 1.6941034604064984, 1.7394100892118591, 2.0203628623309728,
			  1.0608421624361541, 0.50592046426516191, 0.09704373257413107])



header, indices, data = import_csv("DLW_research_runs", indices=2)

for i in range(50, 64):
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
	df.damage_simulation(draws=4000000, peak_temp=temp, disaster_tail=tail, tip_on=on, 
						 temp_map=maps, temp_dist_params=None, maxh=100.0, cons_growth=growth)
	#df.import_damages()

	u = EZUtility(tree=t, damage=df, cost=c, period_len=5.0, eis=eis, ra=ra, time_pref=pref)

	if a <= 2.0:
		ga_model = GeneticAlgorithm(pop_amount=150, num_generations=150, cx_prob=0.8, mut_prob=0.5, 
							bound=2.0, num_feature=63, utility=u, print_progress=True)
		
		gs_model = GradientSearch(learning_rate=0.0001, var_nums=63, utility=u, accuracy=1e-8, 
						  iterations=300, print_progress=True)
		final_pop, fitness = ga_model.run()
		sort_pop = final_pop[np.argsort(fitness)][::-1]
		m_opt, u_opt = gs_model.run(initial_point_list=sort_pop, topk=1)
		
		utility_t, cons_t, cost_t, ce_t = u.utility(m_opt, return_trees=True)
		save_output(m_opt, u, utility_t, cons_t, cost_t, ce_t, prefix=name)
		save_sensitivity_analysis(m_opt, u, utility_t, cons_t, cost_t, ce_t, prefix=name)

	# Constraint first period mitigation to 0.0
	else:
		cfp_m = constraint_first_period(u, 0.0, t.num_decision_nodes)
		cfp_utility_t, cfp_cons_t, cfp_cost_t, cfp_ce_t = u.utility(cfp_m, return_trees=True)
		save_output(cfp_m, u, cfp_utility_t, cfp_cons_t, cfp_cost_t, cfp_ce_t, prefix="CFP_"+name)
		delta_utility = save_sensitivity_analysis(cfp_m, u, cfp_utility_t, cfp_cons_t, cfp_cost_t, cfp_ce_t,
											    "CFP_"+name, return_delta_utility=True)
		delta_utility_x = delta_utility - cfp_utility_t[0]
		save_constraint_analysis(cfp_m, u, delta_utility_x, prefix="CFP_"+name)

#utility_t, cons_t, cost_t, ce_t = u.utility(m, return_trees=True)

#final_pop, fitness = ga_model.run()
#sort_pop = final_pop[np.argsort(fitness)][::-1]

#m_opt, u_opt = gs_model.run(initial_point_list=sort_pop, topk=1)
#m_opt, u_opt = gs_model.run(initial_point_list=[m], topk=1)

#m_opt = sort_pop[0]
#for i in range(63):
#	plot_mitigation_at_node(m, i, u, save=True, prefix="")

#m_opt = m
#m_opt = NodeMaximum.run(m_opt, u)	

#utility_t, cons_t, cost_t, ce_t = u.utility(m_opt, return_trees=True)
#save_output(m_opt, u, utility_t, cons_t, cost_t, ce_t)
#save_sensitivity_analysis(m_opt, u, utility_t, cons_t, cost_t, ce_t)

# Constraint first period mitigation to 0.0
#cfp_m = constraint_first_period(m_opt, u, 0.0)
#cfp_utility_t, cfp_cons_t, cfp_cost_t, cfp_ce_t = u.utility(cfp_m, return_trees=True)
#save_output(cfp_m, u, cfp_utility_t, cfp_cons_t, cfp_cost_t, cfp_ce_t, prefix="CFP")
#save_sensitivity_analysis(cfp_m, u, cfp_utility_t, cfp_cons_t, cfp_cost_t, cfp_ce_t, "CFP")

# everything else in run can easily be created too