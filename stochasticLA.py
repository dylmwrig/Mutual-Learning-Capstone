import random

# stochastic learning automaton with stochastic environment
# my automaton will be given a list of colors and will identify whether it falls under red, green, or blue

# stochastic learning automaton
# start by randomly choosing among available actions
# give the action to the environment and observe the output reward or penalty (represented as 1 and 0 respectively)
# adjust probability that actions associated with reward will be chosen when a reward is given
# do not adjust anything when a penalty is given
# stop on convergence ie action probability of one choice reaches ~ 0.9
class LearningAutomaton:
    def __init__(self, action_set, action_probs, step_size):
        self.action_set = action_set
        self.action_probs = action_probs
        self.step_size = step_size

    def choose_action(self):
        choice = random.random()
        if choice < self.action_probs[0]:
            return self.action_set[0]
        elif choice < self.action_probs[0] + self.action_probs[1]:
            return self.action_set[1]
        else:
            return self.action_set[2]

    # p(n + 1) = pi(n) + step_size(1 - pi(n))
    # then adjust the other action probs so they all add up to 1
    # no need to pass in environment response as this will only be called when the environment returns a reward
    def adjust_probs(self, good_action):
        if good_action=="red":
            reward_index = 0
        elif good_action=="green":
            reward_index = 1
        else:
            reward_index = 2

        old_prob = self.action_probs[reward_index]
        new_prob = old_prob + (self.step_size * (1 - old_prob))
        prob_jump = new_prob - old_prob
        prob_reduction = prob_jump / 2.0
        negative_prob_index = -1 # initialize to negative one to flag when a negative probability is found

        # if the probability becomes so low that it goes into the negative
        # identify the other probability we are trying to reduce and reduce it by that amount again
        # do not change the probability of the element which just tried to go into the negative
        for index, prob in enumerate(self.action_probs):
            if prob - prob_reduction < 0:
                negative_prob_index = index
                prob_reduction = prob_jump

        self.action_probs[reward_index] = new_prob
        for index,prob in enumerate(self.action_probs):
            if index!=reward_index and index!=negative_prob_index:
                reduced_prob = self.action_probs[index] - prob_reduction
                self.action_probs[index] = reduced_prob

# create stochastic environment
# environment must have reward probabilities associated with each action
# environment must output a 1 for a reward and a 0 for a penalty
# reward status decided by choosing a random number and checking if it is smaller than an action's associated reward probability
class Environment:
    def __init__(self, reward_probs):
        self.reward_probs = reward_probs

    def give_feedback(self, action):
        reward_roll = random.random()

        if action=="red":
            if reward_roll < self.reward_probs[0]:
                return 1
        elif action=="green":
            if reward_roll < self.reward_probs[1]:
                return 1
        elif action=="blue":
            if reward_roll < self.reward_probs[2]:
                return 1

        return 0

# run the algorithm 100 times with separate seeds, using the same step size in each
# on convergence (highest action prob >= 0.9) record the winning action and whether or not it is the correct one (red)
# find the percentage of times the algorithm came to the correct conclusion (accuracy)
# also track the average iteration count (speed of convergence)
# repeat the entire experiment using step sizes: 0.01, 0.05, 0.1, 0.2, 0.5
#
# record findings in a table of step-size, accuracy, and speed of convergence
#
# expected results: when step size is increased:
# --speed of convergence increases
# --accuracy decreases


def main():

    # I'm not sure that this is necessary, but I kept it in to illustrate the idea that the LA should converge on red
    color_set = ["Scarlet", "Mahogany", "Vermilion", "Crimson",
                 "Lime", "Emerald",
                 "Prussian"]
    action_set = ["red", "green", "blue"]
    reward_probs = [0.8, 0.4, 0.2]
    step_sizes = [0.01, 0.05, 0.1, 0.2, 0.5]

    environment = Environment(reward_probs)
    iter_count_list = []
    accuracy_list = []

    for step_num, step_size in enumerate(step_sizes):

        correct_choice_count = 0
        run_count = 1
        iteration_count = 1

        while (run_count < 101):
            random.seed(run_count * step_num)

            # initialize all action probabilities to be equal to each other
            action_probs = [0.33, 0.33, 0.33]

            # continue execution until the largest action probability is 0.9
            largest_prob = action_probs[0]
            learning_automaton = LearningAutomaton(action_set, action_probs, step_size)

            while largest_prob < 0.9:
                action = learning_automaton.choose_action()
                reward = environment.give_feedback(action)

                if (reward):
                    learning_automaton.adjust_probs(action)

                # update largest probability to break when one action has a probability of 0.9
                for best_index, probability in enumerate(learning_automaton.action_probs):
                    if probability > largest_prob:
                        largest_prob = probability
                        if best_index == 0:
                            correct_choice_count += 1

                iteration_count += 1

            run_count += 1

        iter_count_list.append(iteration_count/100)
        accuracy_list.append(correct_choice_count/iteration_count)

    print(iter_count_list)
    print(accuracy_list)

    print("The automaton took ", iteration_count, " iterations to complete.")

if __name__ == "__main__":
    main()