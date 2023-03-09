from gpt3 import gpt3

gpt3Controller = gpt3()

inputs = ["As soon as you can.", "Sorry I messed up."]
outputs = ["At you earliest convenience.", "I apologise for my wrongdoings."]

gpt3Controller.save_in_context([inputs, outputs], "I can't stand his temper.", 2)
