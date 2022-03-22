from abc import ABC, abstractmethod

class StateManager(ABC):

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
    def render_state(self, state):
        pass

