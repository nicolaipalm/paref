from paref.interfaces.optimizers.stopping_criteria import StoppingCriteria


class TimeLimitReached(StoppingCriteria):
    def __call__(self, time_limit: float = 100):
        pass