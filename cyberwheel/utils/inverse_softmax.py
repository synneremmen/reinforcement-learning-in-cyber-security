import numpy as np

def softmax(q_values):
	sum_e_x = np.sum([np.exp(x) for x in q_values])
	probs = [np.exp(q)/sum_e_x for q in q_values]
	return np.array(probs)

def invert_softmax(probs):
	logp = np.log(np.array(probs) + 1e-9) # small error to avoid zero division
	c = - np.prod(logp) / len(probs) 
	logits = logp + c 
	return logits

def normalize(values, original_values):
	mean = np.mean(original_values)
	std = np.std(original_values)
	print(mean, std)
	return (values - mean) / std

original_q_values = [1,2,3] 
norm_original_q_values = normalize(original_q_values, original_q_values)
probs = softmax(norm_original_q_values)
print("Norm q values:\t", norm_original_q_values)
invert_array = invert_softmax(probs)
print("Inverted softmax values:\t", invert_array)
action_mapping = {0:1, 1:2, 2:3} # num of new actions given action index (action_index: num_repeats)
arr = []
for i, prob in enumerate(probs):
	arr.extend([prob / (i+1)] * (i+1))
arr = np.array(arr)
values = invert_softmax(arr)
print(values)
print()
print(normalize(values, original_q_values))

# print()

# print("Probs:\t", probs)
# print("New values:\t",invert_array)
# print("Normalized:\t",normalize(invert_array, original_q_values))	

# print()

# extended_probs = [probs[action_idx]/action_mapping[action_idx] for action_idx in action_mapping.keys() for _ in range(action_mapping[action_idx])]
# invert_extend = invert_softmax(extended_probs)
# norm = normalize(invert_extend, original_q_values)
# print("Extended probs:\t", extended_probs)
# print("Inverted extended:\t", invert_extend)
# print("Normalized extended:\t", norm)

# print()

# new_probs_parts = []
# for repeats, prob in zip(action_mapping.values(), probs):
# 	# Split each original probability across its repeated new actions
# 	new_probs_parts.append(np.repeat(prob / repeats, repeats))
# new_probs = np.concatenate(new_probs_parts)
# print(new_probs)

# new_probs_parts = []
# for repeats, prob in zip(action_mapping.values(), probs):
# 	# Split each original probability across its repeated new actions
# 	new_probs_parts.append(np.repeat(prob, repeats))
# new_probs = np.concatenate(new_probs_parts)
# print(new_probs)
# print((new_probs - np.min(new_probs)) / (np.max(new_probs) - np.min(new_probs)))