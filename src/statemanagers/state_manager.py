from abc import ABC, abstractmethod

class StateManager(ABC):
    '''
    Class defining an abstract state manager.
    State managers inheritng from this task should have the ability to translate a state from the perspective of player 2 to player 1.
    '''
    @abstractmethod
    def get_initial_state(self):
        pass

    @abstractmethod
    def get_successor_states(self, state, return_moves):
        pass

    @abstractmethod
    def get_successor(self, state, a):
        pass

    @abstractmethod
    def is_final(self, state):
        pass

    @abstractmethod
    def get_winner(self, state):
        pass

    @abstractmethod
    def get_legal_moves(self, state):
        pass

    @abstractmethod
    def get_action_space(self):
        pass

    @abstractmethod
    def get_action_space_size(self):
        pass

    @abstractmethod
    def get_state_size(self):
        pass

    @abstractmethod
    def flip_state(self, state):
        pass

    @abstractmethod
    def flip_action(self, action, state_was_flipped):
        pass

    @abstractmethod
    def flip_distribution(self, D, state_was_flipped):
        pass

    @abstractmethod
    def get_symmetric_flipped_cases(self, case):
        pass

    @abstractmethod
    def render_state(self, state):
        pass

