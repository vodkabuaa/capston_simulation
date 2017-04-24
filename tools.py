import numpy as np
import seaborn as sns
import csv

################
### Plotting ###
################

markers = (u'o', u'v', u'^', u'<', u'>', u'8', u's', u'p', u'*', u'h', u'H', u'D', u'd')

def plot(y_data, x_data, index=None, y_index=None, title=None, xlabel=None, ylabel=None, 
		 legend=None, vertical_line=None, save=False, file_name=None, points=False):
	import matplotlib.pyplot as plt
	sns.set_style("whitegrid", {"font.family": [u'Bitstream Vera Sans']})
	sns.set_palette("PuBuGn_d")
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)

	if len(y_data.shape)==1:
		ax.plot(x_data, y_data, label=legend)
		if points:
			ax.scatter(x_data, y_data)
	else:
		for i in range(len(y_data)):
			ax.plot(x_data, y_data[i], label=legend[i])
			if points:
				ax.plot(x_data, y_data[i], markers[i])

	ax.xaxis.grid(False)
	if vertical_line is not None:
		ax.axvline(x=vertical_line, linestyle='dashed', linewidth=1, color='black')
	if index is not None:
		ax.set_xticklabels(index)
	if y_index is not None:
		ax.set_yticklabels(y_index)
	if title is not None:
		ax.set_title(title, fontsize='large')
	if xlabel is not None:
		ax.set_xlabel(xlabel, fontsize='large')
	if ylabel is not None:
		ax.set_ylabel(ylabel, fontsize='large')
	if legend is not None:
		plt.legend(fontsize='large', loc='best')
	if save:
		d = find_path(file_name, "plots", ".png")
		plt.savefig(d)
		plt.close(fig)

	plt.show()

def plot_dict(dictionary, title, xlabel, ylabel):
	y_data, x_data = dictionary.items()
	plot(y_data, x_data, title=title, xlabel=xlabel, ylabel=ylabel)

def plot_mitigation_at_node(m, node, utility, save=False, prefix=""):
	m_copy = m.copy()
	x = np.append(np.linspace(0.0, m[node], 25), np.linspace(m[node], m[node]+0.05, 25))
	x = np.unique(x)
	y = np.zeros(len(x))
	for i in range(len(x)):
		m_copy[node] = x[i]
		y[i] = utility.utility(m_copy)
	plot(y, x, title="Utility vs. Mitigation in Node {}".format(node), xlabel="Mitigation", 
		 ylabel="Utility", vertical_line=m[node], save=save, file_name=prefix+"MAT_{}".format(node))

def change_mitigation(m, add_nodes, reduce_nodes, delta):
	m_copy = m.copy()
	cache = set()
	for node in add_nodes:
		m_copy[node] += 0.01
		cache.add(node)
		wec, bec = u.tree.reachable_end_states(node)
		for next_node in range(31+wec, 32+bec):
			if next_node not in cache:
				m_copy[next_node] += delta
				cache.add(next_node)
			else:
				print(next_node)
	for node in reduce_nodes:
		m_copy[node] -= 0.01
		cache.add(node)
		wec, bec = u.tree.reachable_end_states(node)
		for next_node in range(31+wec, 32+bec):
			if next_node not in cache:
				m_copy[next_node] -= delta
				cache.add(next_node)
			else:
				print(next_node)
	return m_copy

def marginal_analysis(m, u, add_nodes, reduce_nodes, delta):
	utility_t, cons_t, cost_t, ce_t = u.utility(m, return_trees=True)
	new_m = change_mitigation(m, add_nodes, reduce_nodes, delta)
	new_utility_t, new_cons_t, new_cost_t, new_ce_t = u.utility(new_m, return_trees=True)
	for period in cost_t.periods:
		new_utility_t.tree[period] -= utility_t[period]
		new_cons_t.tree[period] -= cons_t[period]
		new_cost_t.tree[period] -= cost_t[period]

	return node_eps

def marginal_utility(m, u, add_nodes, reduce_nodes, delta):
	utility_t, cons_t, cost_t, ce_t = u.utility(m, return_trees=True)
	new_m = change_mitigation(m, add_nodes, reduce_nodes, delta)
	new_utility_t, new_cons_t, new_cost_t, new_ce_t = u.utility(new_m, return_trees=True)
	node_eps = SmallStorageTree(utility_t.decision_times)
	for period in node_eps.periods:
		node_eps.tree[period] = (new_utility_t[period]-utility_t[period])
	return node_eps

def marginal_(m, u, add_nodes, reduce_nodes, delta):
	utility_t, cons_t, cost_t, ce_t = u.utility(m, return_trees=True)
	new_m = change_mitigation(m, add_nodes, reduce_nodes, delta)
	new_utility_t, new_cons_t, new_cost_t, new_ce_t = u.utility(new_m, return_trees=True)
	node_eps = SmallStorageTree(utility_t.decision_times)
	for period in node_eps.periods:
		node_eps.tree[period] = (new_utility_t[period]-utility_t[period])
	return node_eps


def plot_first_order_condition(m, node, utility):
	x = np.array([1.0/(10)**i for i in range(1, 11)])
	y = np.zeros(len(x))
	for i in range(len(x)):
		grad, k = utility.partial_grad(m, node, x[i])
		y[i] = grad
	plot(y, x, title="First Order Check for node {}".format(node), xlabel="Delta", 
		 ylabel="Partial deriv. w.r.t. Mitigation")

###########
### I/O ###
###########

def find_path(file_name, directory="data", file_type=".csv"):
	import os
	cwd = os.getcwd()
	if not os.path.exists(directory):
		os.makedirs(directory)
	d = os.path.join(cwd, os.path.join(directory,file_name+file_type))
	return d

def create_file(file_name):
	import os
	d = find_path(file_name)
	if not os.path.isfile(d):
		open(d, 'w').close()
	return d

def file_exists(file_name):
	import os
	d = find_path(file_name)
	return os.path.isfile(d)

def load_csv(file_name, delimiter=';', comment=None):
	d = find_path(file_name)
	pass

def write_columns_csv(lst, file_name, header=[], index=None, start_char=None, delimiter=';', open_as='wb'):
	d = find_path(file_name)
	if index is not None:
		index.extend(lst)
		output_lst = zip(*index)
	else:
		output_lst = zip(*lst)

	with open(d, open_as) as f:
		writer = csv.writer(f, delimiter=delimiter)
		if start_char is not None:
			writer.writerow([start_char])
		if header:
			writer.writerow(header)
		for row in output_lst:
			writer.writerow(row)

def write_columns_to_existing(lst, file_name, header="", delimiter=';'):
	d = find_path(file_name)
	with open(d, 'r') as finput:
			reader = csv.reader(finput, delimiter=delimiter)
			all_lst = []
			row = next(reader)
			nested_list = isinstance(lst[0], list) or isinstance(lst[0], np.ndarray)
			if nested_list:
				lst = zip(*lst)
				row.extend(header)	
			else:
				row.append(header)
			all_lst.append(row)
			n = len(lst)
			i = 0
			for row in reader:
				if nested_list:
					row.extend(lst[i])
				else:
					row.append(lst[i])
				all_lst.append(row)
				i += 1
	with open(d, 'w') as foutput:
			writer = csv.writer(foutput, delimiter=delimiter)
			writer.writerows(all_lst)
			
def append_to_existing(lst, file_name, header="", index=None, delimiter=';', start_char=None):
	write_columns_csv(lst, file_name, header, index, start_char=start_char, delimiter=delimiter, open_as='a')
import csv
def import_csv(file_name, delimiter=';', header=True, indices=None, start_at=0, break_at='\n', ignore=""):
	d = find_path(file_name)
	input_lst = []
	indices_lst = []
	with open(d, 'r') as f:
		reader = csv.reader(f, delimiter=delimiter)
		for _ in range(0, start_at):
			next(reader)
		if header:
			header_row = next(reader)
		for row in reader:
			if row[0] == break_at:
				break
			if row[0] == ignore:
				continue
			if indices:
				input_lst.append(row[indices:])
				indices_lst.append(row[:indices])
			else:
				input_lst.append(row)
	if header and not indices :
		return header_row, np.array(input_lst, dtype="float64")
	elif header and indices:
		return header_row[indices:], indices_lst, np.array(input_lst, dtype="float64")
	return np.array(input_lst, dtype="float64")



##########
### MP ###
##########


def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    if func_name.startswith('__') and not func_name.endswith('__'): #deal with mangled names
        cls_name = cls.__name__.lstrip('_')
        func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.__mro__:
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)