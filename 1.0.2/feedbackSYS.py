"""
feedback naturally require a third neuron to decide!!!!

eventually, this system also decide "who" takes in feedback, or, where to send the feedback to.
  which gets initialized at creation
"""
class Feedback_1:
  def __init__(self,response):
    self.expected = 0
    self.response = response
    # lambda x: 0 if x < 0 else 1



