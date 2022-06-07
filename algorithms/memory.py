from random import randint
from typing import List, Tuple, Dict

from torch import FloatTensor


class Experience:
    action: List[float]
    observation: List[float]
    reward: float
    done: bool

    def __init__(self,
                 action: List[float] = None,
                 observation: Tuple[float] = None,
                 reward: float = None,
                 done: bool = None):
        self.action = list(action)
        self.observation = list(observation)
        self.reward = reward
        self.done = done

    def add(self, exp2):
        if self.action is None:
            self.action = exp2.action
        elif exp2.action != self.action:
            print("Warning: try to add a different action for same Experience.")
        if self.observation is None:
            self.observation = exp2.observation
        elif exp2.observation != self.observation:
            print("Warning: try to add a different observation for same Experience.")
        if self.reward is None:
            self.reward = exp2.reward
        elif exp2.reward != self.reward:
            print("Warning: try to add a different reward for same Experience.")
        if self.done is None:
            self.done = exp2.done
        elif exp2.done != self.done:
            print("Warning: try to add a different done for same Experience.")

    def unwrap(self):
        return self.action, self.observation, self.reward, self.done


class Memory:
    agent_num: int
    # int is agent_id and list[Experience] the corresponding memory
    agent_memory: Dict[int, List[Experience]]
    # index is the step of a global observation
    global_observations: List[Tuple[float]]

    def __init__(self, agent_num):
        self.agent_num = agent_num

        self.agent_memory = {}
        self.global_observations = []

        for i in range(agent_num):
            self.agent_memory[i] = []

    def add_step(self,
                 step: int,
                 agent_id: int = None,
                 action: FloatTensor = None,
                 observation: Tuple[float] = None,
                 global_observation: Tuple[float] = None,
                 reward: float = None,
                 done: bool = None
                 ) -> None:
        # get experience at this step
        old_experience = self.get(agent_id, step)[0]
        new_experience = Experience(action, observation, reward, done)

        if old_experience is not None:  # it was a partial experience
            old_experience.add(new_experience)
            self.agent_memory[agent_id][step] = old_experience
        else:
            self.agent_memory[agent_id].append(new_experience)

        if step == len(self.global_observations):
            self.global_observations.insert(step, global_observation)

    def get(self, agent_num: int, step: int) -> (Experience, List[float]):
        if len(self.agent_memory[agent_num]) == step:
            return None, None
        return self.agent_memory[agent_num][step], self.global_observations[step]

    def get_next_steps(self, steps: List[int]) -> tuple:
        actions = []
        previous_actions = []
        observations = []
        global_observations = []
        rewards = []

        for step in steps:
            next_step = step + 1
            # add empty list in each returned arrays:
            actions.append([])
            previous_actions.append([])
            observations.append([])
            rewards.append([])

            # add experience of each agent at this step
            for agent_id in range(self.agent_num):
                experience, global_observation = self.get(agent_id, next_step)

                action, observation, reward, _ = experience.unwrap()

                exp_previous, _ = self.get(agent_id, next_step - 1)
                previous_action, _, _, _ = exp_previous.unwrap()

                rewards[next_step].append(reward)
                observations[next_step].append(observation)
                previous_actions[next_step].append(previous_action)
                actions[next_step].append(action)

                if next_step > len(global_observations) - 1:
                    global_observations.append(global_observation)

        return (steps,
                rewards,
                global_observations,
                observations,
                previous_actions,
                actions)

    def get_batch(self, num: int) -> tuple:
        actions = []
        previous_actions = []
        observations = []
        global_observations = []
        rewards = []

        # used for having unique step in batch
        steps = []

        for loop in range(num):
            # add empty list in each returned arrays:
            actions.append([])
            previous_actions.append([])
            observations.append([])
            rewards.append([])

            # select a step that is not in the batch
            step = None
            step_in = True
            while step_in:
                step = randint(0, len(self.agent_memory[0]) - 2)  # -2 because, you don't want the last step
                step_in = step in steps
            steps.append(step)

            # add experience of each agent at this step
            for agent_id in range(self.agent_num):
                experience, global_observation = self.get(agent_id, step)

                action, observation, reward, _ = experience.unwrap()
                if step == 0:
                    # since it's first step take a zero action as previous
                    previous_action = [0] * len(action)
                else:
                    exp_previous, _ = self.get(agent_id, step - 1)
                    previous_action, _, _, _ = exp_previous.unwrap()

                rewards[loop].append(reward)
                observations[loop].append(observation)
                previous_actions[loop].append(previous_action)
                actions[loop].append(action)

                if loop > len(global_observations) - 1:
                    global_observations.append(global_observation)

        return (steps,
                rewards,
                global_observations,
                observations,
                previous_actions,
                actions)

    def clear(self):
        for i in range(self.agent_num):
            self.agent_memory[i] = []
