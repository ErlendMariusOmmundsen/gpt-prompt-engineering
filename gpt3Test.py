from gpt3 import gpt3

gpt3Controller = gpt3()

inputs = ["As soon as you can.", "Sorry I messed up."]
outputs = ["At you earliest convenience.", "I apologise for my wrongdoings."]

gpt3Controller.in_context_pipe([inputs, outputs], "I can't stand his temper.", 2)
gpt3Controller.induce_pipe([inputs, outputs], 2)
