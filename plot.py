import numpy as np
import matplotlib.pyplot as plt

def fix(full_trains, full_tests):
	full_tests2 = []
	for train,test in zip(full_trains,full_tests):
		new_test = []
		for i in range(len(train)):
			if i == 0:
				t2 = 0.0
			else:
				tt = train[:i]
				index = tt.index(max(tt))
				t2 = test[index]
			new_test.append(t2)
		full_tests2.append(new_test)
	return full_trains, full_tests

def parse_single(name):
	file = open(name)
	best_trains = []
	best_tests = []
	trains = []
	tests = []
	full_trains = []
	full_tests = []
	for line in file.readlines():
		if '[Iter: 0][Ep: 0.00]' in line:
			trains = []
			tests = []
		if 'Iter' in line:
			train = line.split(' ')[-3]
			test = line.split(' ')[-1]
			# print line, train, test
			test = test[:test.index(']')]
			trains.append(float(train))
			tests.append(float(test))
		if 'Saving' in line:
			best_train = max(trains)
			max_train_index = trains.index(best_train)
			best_test = tests[max_train_index]
			best_trains.append(best_train)
			best_tests.append(best_test)
			full_trains.append(trains)
			full_tests.append(tests)
			trains = []
			tests = []
	print 'train', np.mean(np.array(best_trains)), np.std(np.array(best_trains))
	print 'test', np.mean(np.array(best_tests)), np.std(np.array(best_tests))
	return fix(full_trains, full_tests)

def parse_double(name):
	file = open(name)
	win_trains = []
	win_tests = []
	lose_trains = []
	lose_tests = []
	trains1 = []
	tests1 = []
	trains2 = []
	tests2 = []
	full_trains = []
	full_tests = []
	for line in file.readlines():
		if '[Iter: 0][Ep: 0.00]' in line:
			trains1 = []
			tests1 = []
			trains2 = []
			tests2 = []
		if 'Iter' in line:
			train = line.split(' ')[-3]
			test = line.split(' ')[-1]
			# print line, train, test
			test = test[:test.index(']')]
			if 'Tr1' in line:
				trains1.append(float(train))
				tests1.append(float(test))
			if 'Tr2' in line:
				trains2.append(float(train))
				tests2.append(float(test))
		if 'Saving' in line:
			best_train1 = max(trains1)
			max_train_index1 = trains1.index(best_train1)
			best_test1 = tests1[max_train_index1]
			best_train2 = max(trains2)
			max_train_index2 = trains2.index(best_train2)
			best_test2 = tests2[max_train_index2]
			if best_train1 > best_train2:
				win_trains.append(best_train1)
				win_tests.append(best_test1)
				lose_trains.append(best_train2)
				lose_tests.append(best_test2)
				full_trains.append(trains1)
				full_tests.append(tests1)
			else:
				win_trains.append(best_train2)
				win_tests.append(best_test2)
				lose_trains.append(best_train1)
				lose_tests.append(best_test1)
				full_trains.append(trains2)
				full_tests.append(tests2)
			trains1 = []
			tests1 = []
			trains2 = []
			tests2 = []
	print 'win train', np.mean(np.array(win_trains)), np.std(np.array(win_trains))
	print 'win test', np.mean(np.array(win_tests)), np.std(np.array(win_tests))
	print 'lose train', np.mean(np.array(lose_trains)), np.std(np.array(lose_trains))
	print 'lose test', np.mean(np.array(lose_tests)), np.std(np.array(lose_tests))
	return fix(full_trains, full_tests)

base = 'res/baseline.txt'
base_large = 'res/baseline_large.txt'
base_larger = 'res/baseline_larger.txt'
base_rewards = 'res/baseline_rewards.txt'
r000 = 'res/rs0_do0_ts0_multiple.txt'
r001 = 'res/rs0_do0_ts1_multiple.txt'
r010 = 'res/rs0_do1_ts0_multiple.txt'
r011 = 'res/rs0_do1_ts1_multiple.txt'
r100 = 'res/rs1_do0_ts0_multiple.txt'
r101 = 'res/rs1_do0_ts1_multiple.txt'
r110 = 'res/rs1_do1_ts0_multiple.txt'
r111 = 'res/rs1_do1_ts1_multiple.txt'

def plot(full, c1, name):
	lengths = [len(single) for single in full]
	# max_len = max(lengths)
	max_len = 500
	average = [[] for _ in range(max_len)]
	for single in full:
		while len(single) < max_len:
			best = single[-1]
			single.append(best)
		for (elem, index) in zip(single,range(len(single))):
			average[index].append(elem)
	means = []
	sds = []
	for ave in average:
		means.append(np.mean(np.array(ave)))
		sds.append(np.std(np.array(ave)))
	cutoff = 100 #max_len
	means = np.array(means)[:cutoff]
	sds = np.array(sds)[:cutoff]
	# import pdb; pdb.set_trace()
	# print average
	x = np.array(range(max_len))[:cutoff]
	plt.ylim(0,100)
	# plt.xlim(0,10)
	plt.xlabel('epoch', fontsize=18)
	plt.ylabel('test accuracy (%)', fontsize=18)
	plt.plot(x, means, color=c1, label=name)
	plt.legend(loc='lower right')
	plt.fill_between(x, means-sds, means+sds, 
		alpha=0.1, edgecolor=c1, facecolor=c1,
    	linewidth=1, antialiased=True)

print
print base
print
print base_large
full_trains, full_tests = parse_single(base_large)
# plot(full_tests, 'blue')
print
print base_larger
full_trains, full_tests = parse_single(base_larger)
# plot(full_tests, 'blue')

def v1():
	print base
	full_trains, full_tests = parse_single(base)
	plot(full_tests, 'blue', 'Coop, base')
	print
	print base_rewards
	full_trains, full_tests = parse_single(base_rewards)
	print
	print r000
	full_trains, full_tests = parse_double(r000)
	print
	print r001
	full_trains, full_tests = parse_double(r001)
	plot(full_tests, 'red', 'Comp, TS')
	print
	print r010
	full_trains, full_tests = parse_double(r010)
	# import pdb; pdb.set_trace()
	plot(full_tests, 'black', 'Comp, DO')
	print
	print r011
	full_trains, full_tests = parse_double(r011)
	plot(full_tests, 'green', 'Comp, DO+TS')
	plt.show()

def v2():
	print base
	full_trains, full_tests = parse_single(base)
	plot(full_tests, 'blue', 'Coop, base')
	print
	print r100
	full_trains, full_tests = parse_double(r100)
	print
	print r101
	full_trains, full_tests = parse_double(r101)
	plot(full_tests, 'red', 'Comp, RS+TS')
	print
	print r110
	full_trains, full_tests = parse_double(r110)
	# plot(full_tests, 'green')
	plot(full_tests, 'black', 'Comp, RS+DO')
	print
	print r111
	full_trains, full_tests = parse_double(r111)
	plot(full_tests, 'green', 'Comp, RS+DO+TS')
	# plot(full_tests, 'red')
	plt.show()

def v3():
	print base
	full_trains, full_tests = parse_single(base)
	plot(full_tests, 'blue', 'Coop, base')
	print
	print base_rewards
	full_trains, full_tests = parse_single(base_rewards)
	plot(full_tests, 'navy', 'Coop, rewards')
	print
	print base_large
	full_trains, full_tests = parse_single(base_large)
	plot(full_tests, 'slateblue', 'Coop, params')
	print
	print r000
	full_trains, full_tests = parse_single(r000)
	plot(full_tests, 'cornflowerblue', 'Coop, double')
	print
	print r011
	full_trains, full_tests = parse_double(r011)
	plot(full_tests, 'red', 'Comp, DO+TS')
	plt.show()

v3()











