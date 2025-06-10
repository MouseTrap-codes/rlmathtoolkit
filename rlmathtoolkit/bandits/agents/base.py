from abc import ABC, abstractmethod

class Agent(ABC):
  def __init__(self, n):
    self.n = n 
  
  @abstractmethod
  def select_action(self):
    pass
  
  @abstractmethod
  def update(self, action, reward):
    pass