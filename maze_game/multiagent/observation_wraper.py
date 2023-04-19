from pettingzoo.utils import BaseWrapper


class LastObservationWrapper(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None, return_info=True, options=None):
        super().reset(seed=seed, options=options)

    def last(self, observe: bool = True):
        agent = self.agent_selection
        assert agent
        observation = self.observe(agent) if observe else None
        return (
            observation,
            self.rewards[agent],
            self.terminations[agent],
            self.truncations[agent],
            self.infos[agent],
        )
