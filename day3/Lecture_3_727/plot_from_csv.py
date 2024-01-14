import Lecture_3_727
import matplotlib.pyplot as plt

runner_adam = Lecture_3_727.model_runner("Adam", None, None, None)

runner_adam.read_csv()

plt.rcParams.update({'font.size': 14})

fig = plt.figure(figsize=(16, 8))

Lecture_3_727.plot_model(fig, 1, 1, runner_adam)

# fig.subplots_adjust(hspace=0.5)

plt.show()