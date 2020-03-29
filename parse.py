import numpy as np

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
	return full_trains, full_tests

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
			else:
				lose_trains.append(best_train1)
				lose_tests.append(best_test1)
				win_trains.append(best_train2)
				win_tests.append(best_test2)
			trains1 = []
			tests1 = []
			trains2 = []
			tests2 = []
	print 'win train', np.mean(np.array(win_trains)), np.std(np.array(win_trains))
	print 'win test', np.mean(np.array(win_tests)), np.std(np.array(win_tests))
	print 'lose train', np.mean(np.array(lose_trains)), np.std(np.array(lose_trains))
	print 'lose test', np.mean(np.array(lose_tests)), np.std(np.array(lose_tests))
	return

base = '../old/lang-emerge/res/baseline.txt'
base_vocab16 = '../old/lang-emerge/res/baseline_vocab16.txt'
base_vocab64 = '../old/lang-emerge/res/baseline_vocab64.txt'
base_large = '../old/lang-emerge/res/baseline_large.txt'
base_larger = '../old/lang-emerge/res/baseline_larger.txt'
base_rewards = '../old/lang-emerge/res/baseline_rewards_v2.txt'
r000 = 'res/rs0_do0_ts0_multiple.txt'
r001 = 'res/rs0_do0_ts1_multiple.txt'
r010 = 'res/rs0_do1_ts0_multiple.txt'
r011 = 'res/rs0_do1_ts1_multiple.txt'
r100 = 'res/rs1_do0_ts0_multiple.txt'
r100_2 = 'res/rs1_do0_ts0_multiple_v2.txt'
r100_3 = 'res/rs1_do0_ts0_multiple_v3.txt'
r100_4 = 'res/rs1_do0_ts0_multiple_v4.txt'
r101 = 'res/rs1_do0_ts1_multiple_v2.txt'
r110 = 'res/rs1_do1_ts0_multiple_v2.txt'
r111 = 'res/rs1_do1_ts1_multiple_v3.txt'

vocab16 = 'res/rs1_do1_ts1_large_vocab_16.txt'
vocab64 = 'res/rs1_do1_ts1_large_vocab_64.txt'

# assert False

print
print base
full_trains, full_tests = parse_single(base)
# import pdb; pdb.set_trace()
print
print base_vocab16
parse_single(base_vocab16)
print
print base_vocab64
parse_single(base_vocab64)
print
print base_large
parse_single(base_large)
print
print base_larger
parse_single(base_larger)
print
print base_rewards
parse_single(base_rewards)
print
print r000
parse_double(r000)
print
print r001
parse_double(r001)
print
print r010
parse_double(r010)
print
print r011
parse_double(r011)
print
print r100_2
parse_double(r100_2)
print
print r101
parse_double(r101)
print
print r110
parse_double(r110)
print
print r111
parse_double(r111)
print
print vocab16
parse_double(vocab16)
print
print vocab64
parse_double(vocab64)
print












